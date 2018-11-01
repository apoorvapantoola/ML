
# coding: utf-8

# In[14]:


from PIL import Image
import os
import cv2
import numpy as np
import pytesseract


# In[19]:


#"/Users/phanikumar/Documents/Work/ImageClass/test"
def del_img(filepath):
    count = 0
    face_cascade = cv2.CascadeClassifier('/Users/phanikumar/Documents/Work/ImageClass/haarcascade_frontalface_default.xml')
    low_cascade = cv2.CascadeClassifier('/Users/phanikumar/Documents/Work/ImageClass/haarcascade_lowerbody.xml')
    img_list = os.listdir(filepath)
    for filename in img_list:
        if not filename.startswith('.'):
            join_path = os.path.join(filepath,filename)
            img = cv2.imread(join_path)
            img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 96, 255, cv2.THRESH_BINARY)
            image_final = cv2.bitwise_and(img2gray, img2gray, mask=mask)
            ret, new_img = cv2.threshold(image_final, 96, 255, cv2.THRESH_BINARY)
            tesout = pytesseract.image_to_string(new_img)
            tesout1 = pytesseract.image_to_string(join_path)
            faces = face_cascade.detectMultiScale(img2gray, 1.3, 5)
            low = low_cascade.detectMultiScale(img2gray, 1.1 , 3)
            
            if len(low) or len(faces):
                if tesout or tesout1:
                    os.remove(join_path)
                    count = count+1
            else:
                os.remove(join_path)
                count = count+1
            
    return print("Total pics deleted:",count)


# In[20]:


path = input("enter path: ")


# In[21]:


del_img(path)

