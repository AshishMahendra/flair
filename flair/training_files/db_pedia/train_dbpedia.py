from flair.models import TARSClassifier
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import pickle
import time,logging
log = logging.getLogger("flair")

file = open('train_data_dp.pkl', 'rb')
train=pickle.load(file)
print("Loaded Train Data")
file = open('test_data_dp.pkl', 'rb')
test=pickle.load(file)
print("Loaded Test Data")
train_ds=[]
test_ds=[]
train_ds=SentenceDataset(train)
test_ds=SentenceDataset(test)
print (train_ds[0])
print (test_ds[0])
corpus = Corpus(train=train_ds,test=test_ds)
start_time= time.time()
# 1. load base TARS
tars = TARSClassifier()#.load("tars-base")
# print(tars)
print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")
# 2. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task("dbpedia_bert_data", label_dictionary=corpus.make_label_dictionary(label_type="dbpedia_bert_data"),label_type="dbpedia_bert_data")
# 3. initialize the text classifier trainer with your corpus
trainer = ModelTrainer(tars, corpus)

start_time= time.time()
# start_perf_test()
# 4. train model
data=trainer.train(base_path='taggers/dbpedia_full_bert_big_head_only', # path to store the model artifact
              learning_rate=0.02, 
              mini_batch_size=16, 
              max_epochs=50,
              shuffle=True,
              monitor_train=False,
              train_with_dev =True,
              embeddings_storage_mode="cuda")
# stop_perf_test()
log.info(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")
log.info(data)
