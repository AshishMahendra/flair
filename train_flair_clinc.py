from flair.models import TARSClassifier
from flair.data import Corpus,Sentence
from flair.datasets import SentenceDataset
from flair.trainers import ModelTrainer
import numpy as np
import time

intent= [
    "restaurant_reviews",
    "nutrition_info",
    "account_blocked",
    "oil_change_how",
    "time",
    "weather",
    "redeem_rewards",
    "interest_rate",
    "gas_type",
    "accept_reservations",
    "smart_home",
    "user_name",
    "report_lost_card",
    "repeat",
    "whisper_mode",
    "what_are_your_hobbies",
    "order",
    "jump_start",
    "schedule_meeting",
    "meeting_schedule",
    "freeze_account",
    "what_song",
    "meaning_of_life",
    "restaurant_reservation",
    "traffic",
    "make_call",
    "text",
    "bill_balance",
    "improve_credit_score",
    "change_language",
    "no",
    "measurement_conversion",
    "timer",
    "flip_coin",
    "do_you_have_pets",
    "balance",
    "tell_joke",
    "last_maintenance",
    "exchange_rate",
    "uber",
    "car_rental",
    "credit_limit",
    "oos",
    "shopping_list",
    "expiration_date",
    "routing",
    "meal_suggestion",
    "tire_change",
    "todo_list",
    "card_declined",
    "rewards_balance",
    "change_accent",
    "vaccines",
    "reminder_update",
    "food_last",
    "change_ai_name",
    "bill_due",
    "who_do_you_work_for",
    "share_location",
    "international_visa",
    "calendar",
    "translate",
    "carry_on",
    "book_flight",
    "insurance_change",
    "todo_list_update",
    "timezone",
    "cancel_reservation",
    "transactions",
    "credit_score",
    "report_fraud",
    "spending_history",
    "directions",
    "spelling",
    "insurance",
    "what_is_your_name",
    "reminder",
    "where_are_you_from",
    "distance",
    "payday",
    "flight_status",
    "find_phone",
    "greeting",
    "alarm",
    "order_status",
    "confirm_reservation",
    "cook_time",
    "damaged_card",
    "reset_settings",
    "pin_change",
    "replacement_card_duration",
    "new_card",
    "roll_dice",
    "income",
    "taxes",
    "date",
    "who_made_you",
    "pto_request",
    "tire_pressure",
    "how_old_are_you",
    "rollover_401k",
    "pto_request_status",
    "how_busy",
    "application_status",
    "recipe",
    "calendar_update",
    "play_music",
    "yes",
    "direct_deposit",
    "credit_limit_change",
    "gas",
    "pay_bill",
    "ingredients_list",
    "lost_luggage",
    "goodbye",
    "what_can_i_ask_you",
    "book_hotel",
    "are_you_a_bot",
    "next_song",
    "change_speed",
    "plug_type",
    "maybe",
    "w2",
    "oil_change_when",
    "thank_you",
    "shopping_list_update",
    "pto_balance",
    "order_checks",
    "travel_alert",
    "fun_fact",
    "sync_device",
    "schedule_maintenance",
    "apr",
    "transfer",
    "ingredient_substitution",
    "calories",
    "current_location",
    "international_fees",
    "calculator",
    "definition",
    "next_holiday",
    "update_playlist",
    "mpg",
    "min_payment",
    "change_user_name",
    "restaurant_suggestion",
    "travel_notification",
    "cancel",
    "pto_used",
    "travel_suggestion",
    "change_volume"
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

dataset = load_dataset("clinc_oos","small")


train_ds=[]
test_ds=[]
# get data in balanced format
train_text,train_label,test_text,test_label=split_balanced(dataset['train']["text"],dataset['train']["intent"])

#change the data in flairs required format
for datapoint,class_val in zip(train_text,train_label):
    train_ds.append(Sentence(datapoint.lower()).add_label('clinc_data', intent[class_val].lower()))
train_ds=SentenceDataset(train_ds)
for datapoint,class_val in zip(test_text,test_label):
    test_ds.append(Sentence(datapoint.lower()).add_label('clinc_data', intent[class_val].lower()))
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
tars.add_and_switch_to_new_task("clinc_data", label_dictionary=corpus.make_label_dictionary(label_type="clinc_data"),label_type="clinc_data")
# 3. initialize the text classifier trainer with your corpus
start_time= time.time()

trainer = ModelTrainer(tars, corpus)
# 4. train model
trainer.train(base_path='taggers/clinc_data', # path to store the model artifacts
              learning_rate=0.02, 
              mini_batch_size=10, 
              max_epochs=10,
              shuffle=True,train_with_dev =False,embeddings_storage_mode="cuda")
print(f"\n\nTime taken to complete the model training : {time.time()-start_time}\n\n")