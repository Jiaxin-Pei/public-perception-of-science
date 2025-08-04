import os
import json
from tqdm import tqdm
import pickle
import lzma
import argparse
import bz2
import re
import zstd
import pandas as pd
from collections import defaultdict

df = pd.read_csv('../data/reddit_df.csv')

subreddits = defaultdict(int)#{'AskWomen':0,'AskMen':0,'asktransgender':0, 'AskWomenOver30':0, 'AskMenOver30':0}
domains = {'nature.com':0,'sciencedaily.com':0,'phys.org':0, 'eurekalert.org':0, 'bioengineer.org':0, 'ncbi.nlm.nih.gov':0, 'sciencedirect.com':0, 'scientificamerican.com':0,
          'newscientist.com':0, 'sciencemag.org':0, 'iflscience.com':0, 'livescience.com':0, 'psypost.org':0, 'onlinelibrary.wiley.com':0, 'sciencealert.com':0, 'pnas.org': 0,
          'sciencenews.org':0, 'news.sciencemag.org':0, 'cell.com': 0, 'medicalxpress.com':0}
urls = set(df['url'])
print(len(urls))
input_path = '/shared/2/datasets/reddit-dump-all/RS/'
output_path = '/shared/2/projects/jiaxin/selective-reporting/data/reddit/RS/'
if not os.path.exists(output_path):
    os.mkdir(output_path)

filelist = os.listdir(input_path)
existing_months = set()
selected_filelist = []
for it in filelist:
    if re.search('201[9]-05\.xz',it) and it.split('.')[0] not in existing_months:
        existing_months.add(it.split('.')[0])
        selected_filelist.append(it)
filelist = selected_filelist
#filelist = [it for it in filelist if re.search('2019-\d\d\.',it) and ]
saved_files = os.listdir(output_path)

saved_list = {}
for filename in saved_files:
    saved_list[filename[:-3]] = 0
print(len(filelist))


data_dict = {}
data_list = []
keys = ['author', 'author_fullname','created_utc', 'domain', 'id', 'num_comments', 'score', 'url','over_18', 'permalink','selftext','subreddit', 'title', 'total_awards_received']

qs = 0
count = 0
skipped = 0

for filename in tqdm(filelist):
    #filename = 'RS_2019-05.xz'
    if filename in saved_list:
        skipped += 1
        print('skip', filename)
        print('skipped',skipped)
        continue
    print(filename)
    

    #if re.search('201', filename):
    #    continue

    if re.search('\.xz', filename):
        zipper = lzma
    elif re.search('\.bz2', filename):
        zipper = bz2
    #elif re.search('\.zst', filename):
    #    zipper = zstd
    else:
        continue

    with zipper.open(input_path + filename, 'rt') as load_f:
        #lines = load_f.readlines()
        for line in tqdm(load_f):
            #line = line.readlines()
            try:
                line = json.loads(line)
                #if 'subreddit' not in line:
                #    continue
                
                if line['url'] in urls:
                    new_item = {k:line[k] for k in keys}
                    #new_item = {'id': line['id'], 'title': line['title'], 'author': line['author'],
                    #            'created_utc': line['created_utc'], 'subreddit': line['subreddit']}
                    #data_list.append(json.dumps(new_item))
                    data_list.append(new_item)
                    subreddits[line['subreddit']] += 1
                    
                    #break

            except:
                continue

                
    #for key in subreddits:
    #    print(key, subreddits[key])
        
    #break
    
    print(len(data_list))
    with lzma.open(output_path + filename.split('.')[0] + '.xz', 'wt') as write_f:
        #write_f.write('\n'.join(question_list) + '\n')
        for line in tqdm(data_list):
            write_f.writelines(json.dumps(line) + '\n')