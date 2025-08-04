from tqdm import tqdm
import pandas as pd
from tqdm.notebook import tqdm
#import fasttext
import re
import random
import numpy as np

def read_file(path):
    with open(path) as r:
        lines = r.readlines()
        lines = [it.strip() for it in lines]
    return lines


def write_file(path, lines):

    with open(path, 'w') as w:
        if type(lines) == str:
            w.writelines(lines)
        elif type(lines) == list:
            for line in tqdm(lines):
                w.writelines(str(line) + '\n')
    print('file saved at: ', path)
    
    
def sample_data(df, N, strategy_dict, sampling_key, random_state):
    sampled_df = pd.DataFrame()
    if strategy_dict:
        for key,ratio in strategy_dict.items():
            s_df = df[(df[sampling_key]>=key[0])&(df[sampling_key]<key[1])]
            #print(len(s_df))
            s_df = s_df.sample(min(int(N*ratio),len(s_df)), random_state=random_state)
            sampled_df = pd.concat([sampled_df, s_df])
    else:
        sampled_df = df.sample(min(N,len(df)), random_state=random_state)
    return sampled_df


def sample_data(df, N, strategy_dict, sampling_key, random_state):
    sampled_df = pd.DataFrame()
    if strategy_dict:
        for key,ratio in strategy_dict.items():
            s_df = df[(df[sampling_key]>=key[0])&(df[sampling_key]<key[1])]
            #print(len(s_df))
            s_df = s_df.sample(min(int(N*ratio),len(s_df)), random_state=random_state)
            sampled_df = pd.concat([sampled_df, s_df])
    else:
        sampled_df = df.sample(min(N,len(df)), random_state=random_state)
    return sampled_df


def df2amt_input(df):
    amt_dict = defaultdict(list)
    for i, row in tqdm(df.reset_index().iterrows()):
        #amt_dict['text_%d_A'%i].append(row['text_a'])
        #amt_dict['text_%d_B'%i].append(row['text_b'])
        
        for key in ['text_a' , 'text_b']:
            amt_dict[key.replace('_', '_%s_'%str(i))].append(replace_emoji(row[key], sep=""))
        for key in ['intimacy_a', 'intimacy_b', 'intimacy_diff']:
            amt_dict[key.replace('_', '_%s_'%str(i))].append(row[key])
    return pd.DataFrame(amt_dict)
def replace_emoji(s, sep=''):
    return demoji.replace_with_desc(s, sep=sep).replace('🥸', '&#129400')


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = "../models/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        predictions = self.model.predict(text, k=2) # returns top 2 matching languages
        return predictions
    
    
#remove hashtags and mentions
def remove_hashtags_and_mentions(text):
    return re.sub("@[a-zA-Z0-9_]{1,15}\s|@[a-zA-Z0-9_]{1,15}$|#+[a-zA-Z0-9(_)]{1,}|#+[a-zA-Z0-9(_)]{1,}$", ' ', text).strip()

#split train val test
def split_labels(length,ratio=[0.8,0.1,0.1],seed=0):
    val_len = int(length * (ratio[0]+ratio[1])) - int(length * ratio[0])
    test_len = length - val_len - int(length*ratio[0])
    split_labels = ['train']*int(length*ratio[0]) + ['val']*val_len + ['test']*test_len
    random.seed(seed)
    random.shuffle(split_labels)
    return split_labels

def softmax(x):
    return np.exp(x)/sum(np.exp(x))