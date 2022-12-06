import spacy
from spacy.tokens import DocBin
import pickle


nlp = spacy.blank("en")

#Load data
training_data = pickle.load(open('./data/TrainData.pickle', 'rb'))
testing_data = pickle.load(open('./data/TestData.pickle', 'rb'))

# print(training_data)
# print(training_data[0][0])
# print(training_data[0][1]['entities'])


# training_data = [
#   ("Tokyo Tower is 333m tall.", [(0, 11, "BUILDING")]),
# ]


# the DocBin will store the example documents

db = DocBin()

for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")


db_test = DocBin()
for text, annotations in testing_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db.to_disk("./data/test.spacy")
