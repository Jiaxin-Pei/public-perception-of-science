''' The inference api built for multilingual textual politeness analysis'''


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F
import torch
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import math
from torch import Tensor
import numpy as np
from multiprocessing import Pool, current_process, Queue, Manager
import os

queue = Queue()
def foo(data):
    gpu_id = queue.get()
    #model = models[gpu_id]
    try:
        # run processing on GPU <gpu_id>
        ident = current_process().ident
        print('{}: starting process on GPU {}'.format(ident, gpu_id))

        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)
        #you can easily switch this with other inference objects like topic, sentiment and sexual content
        inti = Estimator(cuda = True)
        res = inti.predict(data)
        print('{}: finished'.format(ident))
        return res
    finally:
        queue.put(gpu_id)
        #print("queue updated")

def chunks(lst, num):
    n = int(len(lst)/num)
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
    
class multi_gpu_inference():
    def __init__(self, gpus = [0], batch_size=256):
        #self.queue = Queue()
        self.gpus = gpus
        self.batch_size = batch_size
        #self.m = Manager()
        #self.queue = self.m.Queue()
        #self.models = {}
        #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
        #os.environ["CUDA_VISIBLE_DEVICES"]=str(gpus)[1:-1]
        print(str(gpus)[1:-1])
        # initialize the queue with the GPU ids
        for gpu_id in gpus:
            #for _ in range(PROC_PER_GPU):
            queue.put(gpu_id)
            #self.models[gpu_id] = PolitenessEstimator(cuda = gpu_id)
    
    def predict(self, data):
        pool = Pool(processes=len(self.gpus))
        inferred = []
        for res in pool.map(foo, chunks(data,len(self.gpus))):
            inferred += res
        pool.close()
        pool.join()
        return inferred


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

class Estimator():
    def __init__(self, tokenizer, model, num_labels, problem_type = "multi_label_classification", cuda = False, batch_size=256):
        
        TOKENIZER = tokenizer
        MODEL = model
        BATCH_SIZE = batch_size
        self.chunk_size = 0.2
        
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
        self.problem_type = problem_type
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels, problem_type=problem_type)
        self.cuda = cuda
        if cuda:
            self.model.cuda()
        
        self.training_args = TrainingArguments(  
            output_dir='./results',                   # output directory
            num_train_epochs=1,                  # total number of training epochs
            per_device_eval_batch_size=BATCH_SIZE,    # batch size for evaluation
        )

    def data_iterator(self, train_x, chunk_size = 500000):
        if chunk_size < 500000:
            chunk_size = 500000
        n_batches = math.ceil(len(train_x) / chunk_size)
        for idx in range(n_batches):
            x = train_x[idx *chunk_size:(idx+1) * chunk_size]
            yield x

    #eval_data is a list of input or a pandas frame
    def prepare_dataset(self, eval_data, max_length = 100):
        if type(eval_data) == list:
            print('start tokenizing %d lines of text'%len(eval_data))
            eval_encodings = self.tokenizer(eval_data, truncation=True, max_length = max_length, padding=True)
            eval_dataset = MyDataset(eval_encodings, [0]*len(eval_data) if self.num_labels == 1 else [[0.0]*self.num_labels]*len(eval_data))
            return eval_dataset
        
        
    def predict(self, eval_data, max_length = 100):      
        
        eval_iterator = self.data_iterator(eval_data, chunk_size=int(len(eval_data)*self.chunk_size))
        eval_preds = []
        
        for x in tqdm(eval_iterator):   
            trainer = Trainer(
                model=self.model,                         # the instantiated 🤗 Transformers model to be trained
                args=self.training_args,                       # training arguments, defined above
            )
            eval_dataset = self.prepare_dataset(x, max_length)
            eval_preds_raw, eval_labels , _ = trainer.predict(eval_dataset)
            if self.num_labels == 1:
                eval_preds += [it[0] for it in eval_preds_raw]
            else:
                if self.problem_type == "multi_label_classification":
                    eval_preds += [it for it in eval_preds_raw]
                else:
                    print(np.argmax(eval_preds_raw, axis=-1))
                    eval_preds += list(np.argmax(eval_preds_raw, axis=-1))
        
        return eval_preds

def arguments():
    parser = ArgumentParser()
    parser.set_defaults(show_path=False, show_similarity=False)
    parser.add_argument('--predict_data_path', default=None)
    parser.add_argument('--text_key', default='text')
    #parser.add_argument('--saving_path', default=None)

    return parser.parse_args()

def read_file(path):
    with open(path) as r:
        lines = r.readlines()
        lines = [it.strip() for it in lines]
    return lines
                  
def write_file(path, lines):
    with open(path, 'w') as w:
        for line in tqdm(lines):
            w.writelines(str(line) + '\n')
    print('file saved at: ', path)
                  
def main():
    args = arguments()
    inti = PolitenessEstimator(cuda = True)
    print('model loaded')
    if args.predict_data_path[-4:] == '.csv':
        data_df = pd.read_csv(args.predict_data_path)
        text = data_df[args.text_key] 
        Politeness_scores = inti.predict(text)
        data_df['Politeness'] = Politeness_scores
        data_df.to_csv(args.predict_data_path.replace('.csv', '_Politeness.csv'), index=False)
    elif args.predict_data_path[-4:] == '.txt':
        text = read_file(args.predict_data_path)
        Politeness_scores = inti.predict(text)
        write_file(args.predict_data_path.replace('.txt', '_Politeness.txt'), Politeness_scores)
        
                  

if __name__ == "__main__":
    main()