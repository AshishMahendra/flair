from flair.models import TARSClassifier
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import pickle
import time,logging
log = logging.getLogger("flair")
import argparse
train_ds=[]
test_ds=[]

def load_data():
    global train_ds,test_ds,dev_ds
    file = open('train_data_agnews', 'rb')
    train=pickle.load(file)
    print("Loaded Train Data")
    file = open('test_data_agnews', 'rb')
    test=pickle.load(file)
    print("Loaded Test Data")
    train_ds=[]
    test_ds=[]
    train_ds=SentenceDataset(train)
    test_ds=SentenceDataset(test)
    print (train_ds[0])
    print (test_ds[0])
def train_module(model,fine_tune,ff_dim,nhead):
    global train_ds,test_ds
    load_data()
    corpus = Corpus(train=train_ds,test=test_ds)
    start_time= time.time()
    # 1. load base TARS
    tars = TARSClassifier()#.load("tars-base")
    # print(tars)
    print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")
    # 2. make the model aware of the desired set of labels from the new corpus
    tars.add_and_switch_to_new_task("ag_news_data", label_dictionary=corpus.make_label_dictionary(label_type="ag_news_data"),label_type="ag_news_data")
    # 3. initialize the text classifier trainer with your corpus
    trainer = ModelTrainer(tars, corpus)

    start_time= time.time()
    # start_perf_test()
    # 4. train model
    data=trainer.train(base_path=f'taggers/agnews_full_bert_big_head_{ff_dim}_{nhead}', # path to store the model artifact
                learning_rate=0.02, 
                mini_batch_size=16, 
                max_epochs=2,
                shuffle=True,
                monitor_train=False,
                train_with_dev =True,
                embeddings_storage_mode="cuda")
    # stop_perf_test()
    log.info(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
    log.info(data)

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-m", "--model", help = "TARS/BERT", default="BERT")
    parser.add_argument("-ft", "--fine_tune", help = "Train the model (True/False)", type=bool,default=False)
    parser.add_argument("-dim", "--ffdim", help = "Feedforward Dimension Size (2048/1024/512/256)",type=int, default=2048)
    parser.add_argument("-nh", "--nhead", help = "Feedforward attention head numbers (8/4/2)", default=8,type=int)

    # Read arguments from command line
    args = parser.parse_args()
    #print(args.model,args.fine_tune,args.ffdim,args.nhead)
    train_module(args.model,args.fine_tune,args.ffdim,args.nhead)