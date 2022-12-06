import os
import numpy
import pandas
import cv2
import pytesseract
from PIL import Image



# cwd = os.getcwd()
# files = os.listdir(cwd)
# print("Files in %r: %s" % (cwd, files))


#pytesseract.pytesseract.tesseract_cmd = r"C:\My Files\Tesseract-OCR\tesseract.exe"

img_cv = cv2.imread(r"C:\My Files\Coding\PyCoding\Udemy Courses\Document Scanner\Cards storage\052.jpeg")
# cv2.imshow("Business Card", img_cv)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(type(img_cv))

img_pl = Image.open(r"C:\My Files\Coding\PyCoding\Udemy Courses\Document Scanner\Cards storage\052.jpeg")
img_pl
type(img_pl)


text_cv=pytesseract.image_to_string(img_cv)
text_cv

text_pl=pytesseract.image_to_string(img_pl)
text_pl

"""text_cv and text_pl have the same results at this stage"""

"""Lesson 15. Image to text to dataframe"""
data = pytesseract.image_to_data(img_cv)
# print(data)
# print(data.split('\n'))



dataList=list(map(lambda x:x.split('\t'), data.split('\n')))
# print(dataList) #возвращает список списков
# for line in dataList:
#     print(line)


df = pandas.DataFrame(dataList[1:], columns=dataList[0]) #возвращает 2-х мерную таблицу
print(df)
# print(df.head())


# """Lesson 16. Clean text in dataframe."""
# df.info()
df.dropna(inplace=True) #drop the missing values
cols = ['level', 'page_num', 'block_num', 'par_num','line_num', 'word_num', 'left', 'top', 'width', 'height', 'conf']
df[cols] = df[cols].astype(float).astype(int)
# print(df.dtypes)
# print("Lesson 16 end")


"""Lesson 17. Draw bounding box around each word"""
img = img_cv.copy()
level = 'word'
for l,x,y,w,h,c, txt in df[['level', 'left', 'top', 'width', 'height', 'conf', 'text']].values:
    # print(l,x,y,w,h,c)
    if level == 'page':
        if l == 1:
            cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,0), 2 )
        else:
            continue
    elif level =='block':
        if l==2:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
        else:
            continue
    elif level == "paragraph":
        if l ==3:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        else:
            continue
    elif level == "line":
        if l ==4:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        else:
            continue
    elif level == "word":
        if l==5:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)
            cv2.putText(img, txt, (x,y), cv2.FONT_HERSHEY_PLAIN,1, (255,0,0),1)
        else:continue

cv2.imshow("boundary box", img)
cv2.waitKey()
cv2.destroyAllWindows()
