from flair.data import Corpus,Sentence
import pickle
intent_list=[
  "World",
  "Sports",
  "Business",
  "Sci/Tech"
]

from datasets import load_dataset
dataset = load_dataset(
   'ag_news')
train_ds=[]
test_ds=[]

for datapoint,class_val in zip(dataset["train"]["text"],dataset["train"]["label"]):
    train_ds.append(Sentence(datapoint.lower()).add_label('ag_news_data', intent_list[class_val].lower()))

with open("train_data_agnews","wb") as f:
    pickle.dump(train_ds, f)

for datapoint,class_val in zip(dataset["test"]["text"],dataset["test"]["label"]):
    test_ds.append(Sentence(datapoint.lower()).add_label('ag_news_data', intent_list[class_val].lower()))

with open("test_data_agnews","wb") as f:
    pickle.dump(test_ds, f)
