import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
import re
import string
import warnings

warnings.filterwarnings('ignore')

# Load NER model
model_ner = spacy.load('./output/model-best/')


def cleanText(txt):
    whitespaces = string.whitespace
    punctuation ="!#$%&\'“”()*+:;<=>?[\\]^`{|}~"
    tableWhitespace = str.maketrans('', '', whitespaces)
    tablePunctuation = str.maketrans('', '', punctuation)
    text = str(txt).lower()
    removeWhitespaces = text.translate(tableWhitespace)
    removePunctuation = removeWhitespaces.translate(tablePunctuation)
    
    return str(removePunctuation)


#group the labels
class GroupGen:
    
    def __init__(self):
        self.id = 0
        self.text = ''
    
    def get_group(self, text):
        if self.text == text:
            return self.id
        else:
            self.id+=1
            self.text = text
            return self.id

grp_gen = GroupGen()


# Parser
def parser(text, label):
    if label =='PHONE':
        text = text.lower()
        text = re.sub(r'\D', '', text)
    elif label == 'EMAIL':
        text = text.lower()
        allow_special_characters = '@_.\-'
        text = re.sub(r'[^A-Za-z10-9{} ]'.format(allow_special_characters), '', text)
    elif label == 'WEB':
        text = text.lower()
        allow_special_characters = ':/.%#\-'
        text = re.sub(r'[^A-Za-z10-9{} ]'.format(allow_special_characters), '', text)
    elif label in ('NAME', 'DES'):
        text = text.lower()
        text = re.sub(r'[^a-z ]', '', text)
        text = text.title()
    elif label == 'ORG':
        text = text.lower()
        text = re.sub(r'[^a-z0-9 ]', '', text)
        text = text.title()
    return text


def getPredictions(image): 
    # etraxt data
    tessData = pytesseract.image_to_data(image)
    #convert into DataFrame
    tessList=list(map(lambda x:x.split('\t'), tessData.split('\n')))
    df = pd.DataFrame(tessList[1:], columns=tessList[0])
    #clean text
    df.dropna(inplace=True)
    df['text']=df['text'].apply(cleanText)

    #convert data into content
    df_clean = df.query('text!="" ')
    content = " ".join([w for w in df_clean['text']])
    #get_prediction from NER model
    doc = model_ner(content)

    #converting doc to json
    docjson=doc.to_json()
    doc_text = docjson['text']

    #creating tokens
    dataframe_tokens = pd.DataFrame(docjson['tokens'])
    dataframe_tokens['token']=dataframe_tokens[['start', 'end']].apply(lambda x:doc_text[x[0]:x[1]], axis=1)
    right_table = pd.DataFrame(docjson['ents'])[['start', 'label']]
    dataframe_tokens = pd.merge(dataframe_tokens, right_table, how='left', on='start')
    dataframe_tokens=dataframe_tokens.fillna('O')

    #join label to df_clean dataframe
    df_clean['end'] = df_clean['text'].apply(lambda x: len(x)+1).cumsum() - 1
    df_clean['start'] = df_clean[['text', 'end']].apply(lambda x:x[1]-len(x[0]), axis=1)

    #inner join with start 
    dataframe_info = pd.merge(df_clean, dataframe_tokens[['start', 'token', 'label']], how='inner', on='start')

    # отметить все распознанные тэги на картинке для иллюстрации
    bb_df = dataframe_info.query("label !='O' ")

    bb_df['label'] = bb_df['label'].apply(lambda x: x[2:])
    bb_df['group'] = bb_df['label'].apply(grp_gen.get_group)

    # right and bottom bounding box
    bb_df[['left', 'top','width', 'height']] = bb_df[['left', 'top', 'width', 'height']].astype(int)
    bb_df['right'] = bb_df['left'] + bb_df['width']
    bb_df['bottom'] = bb_df['top'] + bb_df['height']


    #tagging: groupby group
    col_group = ['left', 'top', 'right', 'bottom', 'label', 'token', 'group']
    group_tag_img = bb_df[col_group].groupby(by='group')
    img_tagging = group_tag_img.agg({
            'left':min,
            'right':max,
            'top':min,
            'bottom':max,
            'label': np.unique,
            'token':lambda x: " ".join(x)
            })

    # отметить все распознанные тэги на картинке для иллюстрации
    img_bb = image.copy()
    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bb,(l,t), (r,b), (0,255,0), 2)
        cv2.putText(img_bb,str(label[0]),(l,t), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255),2)

    # Entities
    info_array = dataframe_info[['token', 'label']].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB=[])
    previous = 'O'

    for token, label in info_array:
        bio_tag = label[:1]
        label_tag = label[2:]
        # parse the token
        text = parser(token, label_tag)

        if bio_tag in ('B', 'I'):
            if previous !=label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] = entities[label_tag][-1]+" "+text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1]+text
        previous = label_tag
    return img_bb, entities
        