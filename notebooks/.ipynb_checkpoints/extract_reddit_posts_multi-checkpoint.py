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
from multiprocessing import Pool, cpu_count

keys = ['author', 'created_utc', 'domain', 'id', 'num_comments', 'score', 'url',
            'over_18', 'permalink', 'selftext', 'subreddit', 'title']

def process_file(filename):
    global urls, data_list, subreddits
    
    if filename in saved_list:
        print('skip', filename)
        return

    print(filename)

    if re.search('\.xz', filename):
        zipper = lzma
    elif re.search('\.bz2', filename):
        zipper = bz2
    else:
        return

    with zipper.open(input_path + filename, 'rt') as load_f:
        for line in tqdm(load_f):
            try:
                line = json.loads(line)
                if line['url'] in urls:
                    new_item = {k: line[k] for k in keys}
                    data_list.append(new_item)
                    subreddits[line['subreddit']] += 1
            except:
                continue

    with lzma.open(output_path + filename.split('.')[0] + '.xz', 'wt') as write_f:
        for line in tqdm(data_list):
            write_f.writelines(json.dumps(line) + '\n')

if __name__ == '__main__':
    #df = pd.read_csv('../data/reddit_df.csv')
    df = pd.read_csv('../data/unique_df_with_scores.csv')
    subreddits = defaultdict(int)
    urls = set(df['url'])
    input_path = '/shared/2/datasets/reddit-dump-all/RS/'
    output_path = '/shared/2/projects/jiaxin/selective-reporting/data/reddit/altmetric_link/'
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    filelist = os.listdir(input_path)
    existing_months = set()
    selected_filelist = []

    for it in filelist:
        if re.search('201[456]-\d\d\.xz', it) and it.split('.')[0] not in existing_months:
            existing_months.add(it.split('.')[0])
            selected_filelist.append(it)

    filelist = selected_filelist
    saved_files = os.listdir(output_path)
    saved_list = {filename[:-3]: 0 for filename in saved_files}

    data_list = []


    # Multiprocessing
    pool = Pool(24)
    pool.map(process_file, filelist)
    pool.close()
    pool.join()
