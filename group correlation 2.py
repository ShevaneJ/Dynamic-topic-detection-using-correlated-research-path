#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import csv
import os
import numpy as np
import spacy
from sklearn.decomposition import PCA
import en_core_web_sm
import pandas as pd
nlp = spacy.load("en_core_web_sm")


# In[2]:


directory ="C:/2003 words prep"
output = "C:/2003 words csv"


# In[3]:


txt_files = os.path.join(directory, '*.txt')


# In[ ]:


array=[]
for txt_file in glob.glob(txt_files):
    with open(txt_file, "rt") as input_file:
        for line in input_file:
            for word in line.split(): 
                array.append(word)
        filename = os.path.splitext(os.path.basename(txt_file))[0] + '.csv'
        
        with open(os.path.join(output, filename), 'w', newline='') as file:
            out_csv = csv.writer(file)
            out_csv.writerow(["word1", "word2", "correlation"])
            for x in array:
                for y in array:
                    #finding vector representation
                    x1=nlp(x)
                    x_vec = np.vstack([word.vector for word in x1 if word.has_vector])
                    y1=nlp(y)
                    y_vec = np.vstack([word.vector for word in y1 if word.has_vector])
                    #finding correlation coefficient
                    corr=np.corrcoef(x_vec, y_vec)[0,1]
                    #writing in csv
                    list=[x,y,corr]
                    out_csv.writerow(list)
                    print(list)
            
      


# In[ ]:




