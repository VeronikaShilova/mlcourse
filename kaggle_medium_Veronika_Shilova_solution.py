import os
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix, hstack
from scipy.stats import probplot
from contextlib import contextmanager
import pickle
from bs4 import BeautifulSoup
import gc
import warnings
#warnings.filterwarnings('ignore')
import time
from html.parser import HTMLParser
from tqdm import tqdm

PATH = '../input/'   # Path to competition data
AUTHOR = 'Veronika_Shilova'  # change here to <name>_<surname>
# it's a nice practice to define most of hyperparams here
SEED = 17
TRAIN_LEN = 62313            # just for tqdm to see progress   
TEST_LEN = 34645             # just for tqdm to see progress
TITLE_NGRAMS = (1, 2)        # for tf-idf on titles
MAX_FEATURES = 50000         # for tf-idf on titles
MEAN_TEST_TARGET = 4.33328   # what we got by submitting all zeros

# nice way to report running times ../input/medium-data/train_log1p_recommends.csv
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def read_json_line(line=None):
    result = None
    try:        
        result = json.loads(line)
    except Exception as e:      
        # Find the offending character index:
        idx_to_replace = int(str(e).split(' ')[-1].replace(')',''))      
        # Remove the offending character:
        new_line = list(line)
        new_line[idx_to_replace] = ' '
        new_line = ''.join(new_line)     
        return read_json_line(line=new_line)
    return result
    
def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def extract_features(path_to_data):
    
    content_list = [] 
    published_list = [] 
    title_list = []
    author_list = []
    domain_list = []
    tags_list = []
    url_list = []
    
    with open(path_to_data, encoding='utf-8') as inp_json_file:
        for line in inp_json_file:
            json_data = read_json_line(line)
            content = json_data['content'].replace('\n', ' \n ').replace('\r', ' \n ') # keep newline
            content_no_html_tags = strip_tags(content)
            content_list.append(content_no_html_tags)
            published = json_data['published']['$date']
            published_list.append(published) 
            title = json_data['meta_tags']['title'].split('\u2013')[0].strip() #'Medium Terms of Service – Medium Policy – Medium'
            title_list.append(title) 
            author = json_data['meta_tags']['author'].strip()
            author_list.append(author) 
            domain = json_data['domain']
            domain_list.append(domain)
            url = json_data['url']
            url_list.append(url)
            
            tags_str = []
            soup = BeautifulSoup(content, 'lxml')
            try:
                tag_block = soup.find('ul', class_='tags')
                tags = tag_block.find_all('a')
                for tag in tags:
                    tags_str.append(tag.text.translate({ord(' '):None, ord('-'):None}))
                tags = ' '.join(tags_str)
            except Exception:
                tags = 'None'
            tags_list.append(tags)
            
    return content_list, published_list, title_list, author_list, domain_list, tags_list, url_list
    
content_list, published_list, title_list, author_list, domain_list, tags_list, url_list = \
                                                                            extract_features(os.path.join(PATH, 'train.json'))
train = pd.DataFrame()
train['content'] = content_list
train['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
train['title'] = title_list
train['author'] = author_list
train['domain'] = domain_list
train['tags'] = tags_list
train['url'] = url_list

content_list, published_list, title_list, author_list, domain_list, tags_list, url_list = \
                                                                            extract_features(os.path.join(PATH, 'test.json'))

test = pd.DataFrame()
test['content'] = content_list
test['published'] = pd.to_datetime(published_list, format='%Y-%m-%dT%H:%M:%S.%fZ')
test['title'] = title_list
test['author'] = author_list
test['domain'] = domain_list
test['tags'] = tags_list
test['url'] = url_list

train['target'] = pd.read_csv(os.path.join(PATH, 'train_log1p_recommends.csv'), index_col='id').values
y_train = train['target']

del content_list, published_list, title_list, author_list, domain_list, tags_list, url_list

with timer('Tf-Idf for titles'):
    # acticle titles
    tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000)
    X_train_title_sparse = tfidf.fit_transform(train['title'])
    X_test_title_sparse = tfidf.transform(test['title'])
    
with timer('Tf-Idf for content'):
    # acticle content
   tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=50000)
   X_train_content_sparse = tfidf.fit_transform(train['content'])
   X_test_content_sparse = tfidf.transform(test['content'])

   X_train_sparse = hstack([X_train_title_sparse, X_train_content_sparse]).tocsr()
   X_test_sparse = hstack([X_test_title_sparse, X_test_content_sparse]).tocsr()
   full_df = pd.concat([train.drop('target', axis=1), test])
   # Index to split the training and test data sets
   idx_split = train.shape[0]
   full_feat = pd.DataFrame(index=full_df.index)    

def add_time_features(X_train, X_test, full_df):
    scaler = StandardScaler()
    hour = full_df['published'][:idx_split].apply(lambda ts: ts.hour)
    morning = scaler.fit_transform(((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1))
    day = scaler.fit_transform(((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1))
    evening = scaler.fit_transform(((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1))
    night = scaler.fit_transform(((hour >= 0) & (hour <=6)).astype('int').values.reshape(-1, 1))
    
    
    objects_to_hstack_train = [X_train, morning, day, evening, night]
    X_train_sparse = hstack(objects_to_hstack_train).tocsr()
    
    hour = full_df['published'][idx_split:].apply(lambda ts: ts.hour)
    morning = scaler.transform(((hour >= 7) & (hour <= 11)).astype('int').values.reshape(-1, 1))
    day = scaler.transform(((hour >= 12) & (hour <= 18)).astype('int').values.reshape(-1, 1))
    evening = scaler.transform(((hour >= 19) & (hour <= 23)).astype('int').values.reshape(-1, 1))
    night = scaler.transform(((hour >= 0) & (hour <=6)).astype('int').values.reshape(-1, 1))
    
    
    objects_to_hstack_test = [X_test, morning, day, evening, night]
    X_test_sparse = hstack(objects_to_hstack_test).tocsr()
    
    return X_train_sparse, X_test_sparse    

with timer('Preparing time features'):
    X_train_sparse, X_test_sparse = add_time_features(X_train_sparse, X_test_sparse, full_df)

def train_and_predict(X_train_sparse, X_test_sparse, y_train):
    clf = Ridge(alpha=1.0)
    clf.fit(X_train_sparse, y_train)
    y_pred = clf.predict(X_test_sparse)
    return y_pred

with timer('Encoding for authors'):
    X_author_sparse = pd.get_dummies(full_df['author'])
    X_train_author_sparse = X_author_sparse[:idx_split]
    X_test_author_sparse = X_author_sparse[idx_split:]
    X_train_sparse = hstack([X_train_sparse, X_author_sparse[:idx_split],]).tocsr()
    X_test_sparse = hstack([X_test_sparse, X_author_sparse[idx_split:]]).tocsr()
    
with timer('Tf-Idf for tags'):
    tfidf = TfidfVectorizer(ngram_range=(1,6), max_features=90000)
    X_train_tags_sparse = tfidf.fit_transform(train['tags'].fillna('0'))
    X_test_tags_sparse = tfidf.transform(test['tags'].fillna('0'))
    X_train_sparse_with_tags = hstack([X_train_sparse,
                         tfidf.fit_transform(train['tags'].fillna('0'))]).tocsr()
    X_test_sparse_with_tags = hstack([X_test_sparse,
                         tfidf.transform(test['tags'].fillna('0'))]).tocsr()

with timer('Adding day_of_week feature'):
    scaler = StandardScaler()
    dow_train = full_df['published'][:idx_split]
    X_train_dow_sparse = scaler.fit_transform(dow_train.apply(lambda ts: ts.date().weekday()).astype('int').values.reshape(-1,1))
    dow_test = full_df['published'][idx_split:]
    X_test_dow_sparse = scaler.transform(dow_test.apply(lambda ts: ts.date().weekday()).astype('int').values.reshape(-1,1))
    X_train_sparse = hstack([X_train_sparse_with_tags, X_train_dow_sparse]).tocsr()
    X_test_sparse = hstack([X_test_sparse_with_tags, X_test_dow_sparse]).tocsr()

full_feat['length'] = full_df['content'].apply(len)
full_feat['month'] = full_df['published'].apply(lambda x: x.month)

with timer('Adding length feature'):
    X_train_length_sparse = scaler.fit_transform(full_feat['length'][:idx_split].values.reshape(-1,1))
    X_test_length_sparse = scaler.transform(full_feat['length'][idx_split:].values.reshape(-1,1))
    X_train_sparse = hstack([X_train_sparse, X_train_length_sparse]).tocsr()
    X_test_sparse = hstack([X_test_sparse, X_test_length_sparse]).tocsr()
    
with timer('Adding month feature'):
    X_train_month_sparse = scaler.fit_transform(full_feat['month'][:idx_split].values.reshape(-1,1))
    X_test_month_sparse = scaler.transform(full_feat['month'][idx_split:].values.reshape(-1,1))
    X_train_sparse = hstack([X_train_sparse, X_train_month_sparse]).tocsr()
    X_test_sparse = hstack([X_test_sparse, X_test_month_sparse]).tocsr()
    
with timer('Addding domain'):
    X_domain_sparse = pd.get_dummies(full_df['domain'])
    X_train_domain_sparse = X_domain_sparse[:idx_split]
    X_test_domain_sparse = X_domain_sparse[idx_split:]
    X_train_sparse = hstack([X_train_sparse, 
                         X_train_domain_sparse]).tocsr()
    X_test_sparse = hstack([X_test_sparse, 
                         X_test_domain_sparse]).tocsr()

with timer('Ridge: train and predict'):
    ridge_pred = train_and_predict(X_train_sparse, X_test_sparse, y_train)

with timer('Prepare submission'):
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'), index_col='id')
    ridge_pred = ridge_pred + MEAN_TEST_TARGET - y_train.mean()
    submission_df['log_recommends'] = ridge_pred
    submission_df.to_csv(f'submission_medium_{AUTHOR}.csv')