import pandas as pd
import string


with open('businessCard.txt', mode='r', encoding='utf8', errors='ignore') as f:
    text = f.read()

data=list(map(lambda x:x.split('\t'), text.split('\n')))
# print(data)

df = pd.DataFrame(data[1:], columns=data[0])
# print(df.head(30))

whitespaces = string.whitespace
# print(repr(whitespaces))

punctuation ="!#$%&\'“”()*+:;<=>?[\\]^`{|}~"
# print(punctuation)

tableWhitespace = str.maketrans('', '', whitespaces)
tablePunctuation = str.maketrans('', '', punctuation)


def cleanText(txt):
    text = str(txt)
    text = text.lower()
    removeWhitespaces = text.translate(tableWhitespace)
    removePunctuation = removeWhitespaces.translate(tablePunctuation)
    return str(removePunctuation)


df['text'] = df['text'].apply(cleanText)
# print(df.head(30))
dataClean = df.query("text !='' ")
dataClean.dropna(inplace=True)
# print(dataClean.head(30))

group = dataClean.groupby(by='id')
# print(group) #<pandas.core.groupby.generic.DataFrameGroupBy object at 0x00000258FF9CDBD0>

# print(group.groups.keys())


# """Lesson 27. Convert Data into spacy format for all business card"""

def init_cards():
    cards = group.groups.keys()
    allCardsData = []
    for card in cards:
        cardData = []
        grouparray = group.get_group(card)[['text', 'tag']].values
        content = ''
        annotations = {'entities':[]}
        start = 0
        end = 0
        for text, label in grouparray:
            text = str(text)
            stringLen = len(text)+1

            start = end
            end = start+stringLen

            if label !='O':
                annot = (start, end-1,label)
                annotations['entities'].append(annot)

            content = content+text+' '

        cardData=(content, annotations)
        # print(cardData)
        allCardsData.append(cardData)
    return allCardsData


"""Lesson 28. Splitting data into training and testing set"""

def splitting_data():
    import pickle
    TrainData = init_cards()[:240]
    TestData = init_cards()[240:0]
    pickle.dump(TrainData, open('data/TrainData.pickle', mode='wb'))
    pickle.dump(TestData, open('data/TestData.pickle', mode='wb'))

""""""
splitting_data()

