import numpy as np
import pandas as pd
import cv2
import pytesseract

import os
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


imagePaths = glob("../../Storage/*.jpeg")
# print(imagePaths)


imgPath = imagePaths[0] #../../Storage\000.jpeg
# print(imgPath) 

lst = os.path.split(imgPath)
# print(lst)

image = cv2.imread(imgPath)
# print(image)

data = pytesseract.image_to_data(image)
# print(data)

dataList = list(map(lambda x:x.split('\t'), data.split('\n')))
# print(dataList)

df = pd.DataFrame(dataList[1:], columns=dataList[0])
# print(df)

df.dropna(inplace=True)
# print(df)

# print(df['conf'])

df['conf'] = df['conf'].astype(float).astype(int)
# print(df['conf'])

usefuldata = df.query('conf>=30')
# print(usefuldata)


def create_cvfile():
    allBussinessCard = pd.DataFrame(columns=['id', 'text'])
    for imgPath in tqdm(imagePaths, desc='BusinessCard'):
        _, filename = os.path.split(imgPath) # print(filenane) #('../../Storage', '000.jpeg')
        image = cv2.imread(imgPath)
        # print(image)
        data = pytesseract.image_to_data(image)
        dataList = list(map(lambda x:x.split('\t'), data.split('\n')))
        # print(dataList)
        df = pd.DataFrame(dataList[1:], columns=dataList[0])
        # print(df)
        df.dropna(inplace=True)
        # print(df) after dropping values   
        df['conf']=df['conf'].astype(float).astype(int)
        # print(df['conf']) # selects conf column
        useFulData = df.query('conf>=30')

        # Dataframe 
        businessCard = pd.DataFrame()
        # print(businessCard)
        businessCard['text'] = useFulData['text']
        businessCard['id'] = filename
        # print(businessCard)

        #concatenation
        allBussinessCard = pd.concat((allBussinessCard, businessCard))

        #saving to file
        allBussinessCard.to_csv('businessCard.csv', index=False)
        