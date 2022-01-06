from flair.data import Corpus,Sentence
import pickle
intent_list=[
  "Company",
  "EducationalInstitution",
  "Artist",
  "Athlete",
  "OfficeHolder",
  "MeanOfTransportation",
  "Building",
  "NaturalPlace",
  "Village",
  "Animal",
  "Plant",
  "Album",
  "Film",
  "WrittenWork"
]
import pickle
from datasets import load_dataset
dataset = load_dataset(
   'dbpedia_14')
train_ds=[]
test_ds=[]

for datapoint,class_val in zip(dataset["train"]["content"],dataset["train"]["label"]):
    train_ds.append(Sentence(datapoint.lower()).add_label('dbpedia_bert_data', intent_list[class_val].lower()))
with open("train_data_dp.pkl","wb") as f:
    pickle.dump(train_ds, f)
for datapoint,class_val in zip(dataset["test"]["content"],dataset["test"]["label"]):
    test_ds.append(Sentence(datapoint.lower()).add_label('dbpedia_bert_data', intent_list[class_val].lower()))
with open("test_data_dp.pkl","wb") as f:
    pickle.dump(test_ds, f)
