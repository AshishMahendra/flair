from flair.models import TARSClassifier
from flair.data import Sentence

# 5. Load the trained model
path_to_model="taggers/clinc_data"
tars = TARSClassifier.load(f'{path_to_model}/final-model.pt')
# 6. Prepare a test sentence
sentence = Sentence("how would they say butter in zambia")
# 7. Predict 
tars.predict(sentence)
print(sentence)

# # Load zero-shot TARS model
# tars = TARSClassifier.load('tars-base')
# # Prepare a test sentence
# sentence = Sentence("I'm going for a vacation in india")
# labels= ["Holiday","Weather","Play Music"]

# tars.predict_zero_shot(sentence, labels,multi_label=True)

# tars.predict(sentence)
# print(sentence.to_dict())