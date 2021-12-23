from contextlib import ExitStack
from itertools import count
from flair.models import TARSClassifier
from flair.data import Corpus,Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,confusion_matrix
import random
import numpy as np
import time

label_list=[
"bookrestaurant",
"requestride" ,
"getplacedetails",
"getdirections",
"compareplaces",
"sharecurrentlocation" ,
"searchplace",
"shareeta",
"getweather",
"gettrafficinformation",
]

def split_balanced(data, target, test_size=0.2):

    classes = np.unique(target)
    # can give test_size as fraction of input data size of number of samples
    if test_size<1:
        n_test = np.round(len(target)*test_size)
    else:
        n_test = test_size
    n_train = max(0,len(target)-n_test)
    n_train_per_class = max(1,int(np.floor(n_train/len(classes))))
    n_test_per_class = max(1,int(np.floor(n_test/len(classes))))

    ixs = []
    for cl in classes:
        if (n_train_per_class+n_test_per_class) > np.sum(target==cl):
            # if data has too few samples for this class, do upsampling
            # split the data to training and testing before sampling so data points won't be
            #  shared among training and test data
            splitix = int(np.ceil(n_train_per_class/(n_train_per_class+n_test_per_class)*np.sum(target==cl)))
            ixs.append(np.r_[np.random.choice(np.nonzero(target==cl)[0][:splitix], n_train_per_class),
                np.random.choice(np.nonzero(target==cl)[0][splitix:], n_test_per_class)])
        else:
            ixs.append(np.random.choice(np.nonzero(target==cl)[0], n_train_per_class+n_test_per_class,
                replace=False))

    # take same num of samples from all classes
    ix_train = np.concatenate([x[:n_train_per_class] for x in ixs])
    ix_test = np.concatenate([x[n_train_per_class:(n_train_per_class+n_test_per_class)] for x in ixs])
    #print(data[list(ix_test)])
    X_test=[]
    y_test=[]
    X_train=[]
    y_train=[]
    # seperate train & test data on basis of index
    for val in list(ix_test):
        X_test.append(data[val])
        y_test.append(target[val])
    for val in list(ix_train):
        X_train.append(data[val])
        y_train.append(target[val])
    return X_train,y_train,X_test,y_test

from datasets import list_datasets, load_dataset, list_metrics, load_metric


dataset = load_dataset("snips_built_in_intents")
train_ds=[]
test_ds=[]
# get data in balanced format
train_text,train_label,test_text,test_label=split_balanced(dataset['train']["text"],dataset['train']["label"])

#change the data in flairs required format
for datapoint,class_val in zip(train_text,train_label):
    train_ds.append(Sentence(datapoint.lower()).add_label('snips_data_new', label_list[class_val].lower()))
train_ds=SentenceDataset(train_ds)

for datapoint,class_val in zip(test_text,test_label):
    test_ds.append(Sentence(datapoint.lower()).add_label('snips_data_new', label_list[class_val].lower()))
test_ds=SentenceDataset(test_ds)
# print (train_ds[0])
# print (test_ds[0])
corpus = Corpus(train=train_ds,test=test_ds)
start_time= time.time()
# 1. load base TARS
tars = TARSClassifier()
# print(tars)
print(f"\n\nTime taken to load the model : {time.time()-start_time}\n\n")
# 2. make the model aware of the desired set of labels from the new corpus
tars.add_and_switch_to_new_task("snips_data_new", label_dictionary=corpus.make_label_dictionary(label_type="snips_data_new"),label_type="snips_data_new")
# 3. initialize the text classifier trainer with your corpus
start_time= time.time()

trainer = ModelTrainer(tars, corpus)
# 4. train model
trainer.train(base_path='taggers/snips_data_new', # path to store the model artifacts
              learning_rate=0.02, 
              mini_batch_size=10, 
              max_epochs=10,
              shuffle=True,
              train_with_dev = False,
              embeddings_storage_mode="cuda")
print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")