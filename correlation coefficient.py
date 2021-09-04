#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import spacy
from sklearn.decomposition import PCA


# In[2]:


import en_core_web_sm


# In[3]:


nlp = spacy.load("en_core_web_sm")


# In[4]:


x="paper"
x1="superworldsheet"
x2="semilightcone"
x3="simplifies"
x4="involving"
x5="ebigdataprojectrunfileskeywordstextfilesvtxt"
x6="concluding"
x7="remarks"
x8="putting"
x9="heterotic"


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()


# In[ ]:





# In[5]:


x_t=nlp(x)
x_vec = np.vstack([word.vector for word in x_t if word.has_vector])
print(x_vec)


# In[9]:


x1_t=nlp(x1)
x1_vec = np.vstack([word.vector for word in x1_t if word.has_vector])


# In[10]:


x2_t=nlp(x2)
x2_vec = np.vstack([word.vector for word in x2_t if word.has_vector])


# In[11]:


x3_t=nlp(x3)
x3_vec = np.vstack([word.vector for word in x3_t if word.has_vector])


# In[12]:


x4_t=nlp(x4)
x4_vec = np.vstack([word.vector for word in x4_t if word.has_vector])


# In[13]:


x5_t=nlp(x5)
x5_vec = np.vstack([word.vector for word in x5_t if word.has_vector])


# In[14]:


x6_t=nlp(x6)
x6_vec = np.vstack([word.vector for word in x6_t if word.has_vector])


# In[15]:


x7_t=nlp(x7)
x7_vec = np.vstack([word.vector for word in x7_t if word.has_vector])


# In[16]:


x8_t=nlp(x8)
x8_vec = np.vstack([word.vector for word in x8_t if word.has_vector])


# In[17]:


x9_t=nlp(x9)
x9_vec = np.vstack([word.vector for word in x9_t if word.has_vector])


# In[18]:


import csv


# In[19]:


c1=np.corrcoef(x_vec, x1_vec)[0,1]
print(c1)


# In[20]:


c2=np.corrcoef(x_vec, x2_vec)[0,1]


# In[21]:


c3=np.corrcoef(x_vec, x3_vec)[0,1]


# In[22]:


c4=np.corrcoef(x_vec, x4_vec)[0,1]


# In[23]:


c5=np.corrcoef(x_vec, x5_vec)[0,1]


# In[24]:


c6=np.corrcoef(x_vec, x6_vec)[0,1]


# In[25]:


c7=np.corrcoef(x_vec, x7_vec)[0,1]


# In[26]:


c8=np.corrcoef(x_vec, x8_vec)[0,1]


# In[27]:



c9=np.corrcoef(x_vec, x9_vec)[0,1]


# In[28]:


c10=np.corrcoef(x1_vec, x2_vec)[0,1]


# In[29]:


c11=np.corrcoef(x1_vec, x3_vec)[0,1]


# In[30]:


c12=np.corrcoef(x1_vec, x4_vec)[0,1]


# In[31]:


c13=np.corrcoef(x1_vec, x5_vec)[0,1]


# In[32]:


c14=np.corrcoef(x1_vec, x6_vec)[0,1]


# In[33]:


c15=np.corrcoef(x1_vec, x7_vec)[0,1]


# In[34]:


c16=np.corrcoef(x1_vec, x8_vec)[0,1]


# In[35]:


c17=np.corrcoef(x1_vec, x9_vec)[0,1]


# In[36]:


c18=np.corrcoef(x2_vec, x3_vec)[0,1]


# In[37]:


c19=np.corrcoef(x2_vec, x4_vec)[0,1]


# In[38]:


c20=np.corrcoef(x2_vec, x5_vec)[0,1]


# In[39]:


c21=np.corrcoef(x2_vec, x6_vec)[0,1]


# In[40]:


c22=np.corrcoef(x2_vec, x7_vec)[0,1]


# In[41]:


c23=np.corrcoef(x2_vec, x8_vec)[0,1]


# In[42]:


c24=np.corrcoef(x2_vec, x9_vec)[0,1]


# In[43]:


c25=np.corrcoef(x3_vec, x4_vec)[0,1]


# In[44]:


c26=np.corrcoef(x3_vec, x5_vec)[0,1]


# In[45]:


c27=np.corrcoef(x3_vec, x6_vec)[0,1]


# In[46]:


c28=np.corrcoef(x3_vec, x7_vec)[0,1]


# In[47]:


c29=np.corrcoef(x3_vec, x8_vec)[0,1]


# In[48]:


c30=np.corrcoef(x3_vec, x9_vec)[0,1]


# In[49]:


c31=np.corrcoef(x4_vec, x5_vec)[0,1]


# In[50]:


c32=np.corrcoef(x4_vec, x6_vec)[0,1]


# In[51]:


c33=np.corrcoef(x4_vec, x7_vec)[0,1]


# In[52]:


c34=np.corrcoef(x4_vec, x8_vec)[0,1]


# In[53]:


c35=np.corrcoef(x4_vec, x9_vec)[0,1]


# In[54]:


c36=np.corrcoef(x5_vec, x6_vec)[0,1]


# In[55]:


c37=np.corrcoef(x5_vec, x7_vec)[0,1]


# In[56]:


c38=np.corrcoef(x5_vec, x8_vec)[0,1]


# In[57]:


c39=np.corrcoef(x5_vec, x9_vec)[0,1]


# In[58]:


c40=np.corrcoef(x6_vec, x7_vec)[0,1]


# In[59]:


c41=np.corrcoef(x6_vec, x8_vec)[0,1]


# In[60]:


c42=np.corrcoef(x6_vec, x9_vec)[0,1]


# In[61]:


c43=np.corrcoef(x7_vec, x8_vec)[0,1]


# In[62]:


c44=np.corrcoef(x7_vec, x9_vec)[0,1]


# In[63]:


c45=np.corrcoef(x8_vec, x9_vec)[0,1]


# In[73]:


import csv
with open('tagged.csv', 'w') as file:
    writer = csv.writer(file, delimiter = ' ')
    writer.writerow(["coeff"])
    writer.writerow([c1])
    writer.writerow([c2])
    writer.writerow([c3])
    writer.writerow([c4])
    writer.writerow([c5])
    writer.writerow([c6])
    writer.writerow([c7])
    writer.writerow([c8])
    writer.writerow([c9])
    writer.writerow([c10])
    writer.writerow([c11])
    writer.writerow([c12])
    writer.writerow([c13])
    writer.writerow([c14])
    writer.writerow([c15])
    writer.writerow([c16])
    writer.writerow([c17])
    writer.writerow([c18])
    writer.writerow([c19])
    writer.writerow([c20])
    writer.writerow([c21])
    writer.writerow([c22])
    writer.writerow([c23])
    writer.writerow([c24])
    writer.writerow([c25])
    writer.writerow([c26])
    writer.writerow([c27])
    writer.writerow([c28])
    writer.writerow([c29])
    writer.writerow([c30])
    writer.writerow([c31])
    writer.writerow([c32])
    writer.writerow([c33])
    writer.writerow([c34])
    writer.writerow([c35])
    writer.writerow([c36])
    writer.writerow([c37])
    writer.writerow([c38])
    writer.writerow([c39])
    writer.writerow([c40])
    writer.writerow([c41])
    writer.writerow([c42])
    writer.writerow([c43])
    writer.writerow([c44])
    writer.writerow([c45])
    
        

    
    
    


# In[74]:


import pandas as pd


# In[75]:


data = pd.read_csv('tagged.csv', encoding= 'unicode_escape')
data.head()


# In[80]:


corr = { 'coeff': ['0.555438522096989', '0.39915297551154044', '0.5100337110715427', '0.30388364748194846', '0.4446020150869289','0.48464573261388766','0.39131009827734514','0.34534837005627267','0.4761718826241783','0.39631928692943746','0.4556056072775252','0.2338356598561739','0.6039209277732653','0.43724156283403354','0.453327334761969','0.32484986340300526','0.46169781283654887','0.36791303347728316','0.26604902326301655','0.44977452922579236','0.32562208263155556','0.41162383217554604','0.2987149451581722','0.38907756123996906','0.012418598642892138','0.38334077595222077','0.15458340401698978','0.724325246764699','0.08048932118688779','0.44404555739787416','0.3608750400728597','0.7202054714283602','0.11543886080020775','0.8080706415062092','0.13342573587837875','0.42510204099414584','0.3476238899288446','0.32368348401126024','0.40060987500794987','0.19628263982463318','0.7546516802652873','0.29680612889069174','0.29680612889069174','0.31040279572017554','0.14697432676572128']} 


# In[81]:


df = pd.DataFrame(corr)


# In[82]:


print(df)


# In[83]:


df['coeff_Rank'] = df['coeff'].rank(ascending = 1) 


# In[84]:


df = df.set_index('coeff_Rank') 
print(df) 


# In[ ]:




