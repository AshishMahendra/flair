from flair.models import TARSClassifier
from flair.data import Corpus
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import time,pickle

train_ds=[]
file = open('train_data_clinc.pkl', 'rb')
train=pickle.load(file)
print("Loaded Train Data")
train_ds=SentenceDataset(train)
print(train_ds[0])
test_ds=[]
file = open('test_data_clinc.pkl', 'rb')
test=pickle.load(file)
print("Loaded Test Data")
test_ds=SentenceDataset(test)
print(test_ds[0])
dev_ds=[]
file = open('dev_data_clinc.pkl', 'rb')
dev=pickle.load(file)
print("Loaded Dev Data")
dev_ds=SentenceDataset(dev)
print(dev_ds[0])

corpus = Corpus(train=train_ds,test=test_ds,dev=dev_ds)
start_time= time.time()

tars = TARSClassifier()#.load("tars-base")

print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")

tars.add_and_switch_to_new_task("clinc_data", label_dictionary=corpus.make_label_dictionary(label_type="clinc_data"),label_type="clinc_data")

trainer = ModelTrainer(tars, corpus)

start_time= time.time()

data= trainer.train(base_path='taggers/clinc_small_tars_big_head_only', # path to store the model artifacts
              learning_rate=0.02,
              mini_batch_size=16,
              max_epochs=50,
              monitor_train=False, # if we want to monitor train change to True
              embeddings_storage_mode="cuda",
              train_with_dev =True
              )

print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")

print(data)
