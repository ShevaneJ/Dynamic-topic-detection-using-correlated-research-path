#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import numpy as np
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import networkx as nx
import os
from os import listdir
from os.path import isfile, join
import re
import sys
import math


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[3]:


def find_similar(a,b):
vectorizer = TfidfVectorizer()
 tfidf = vectorizer.fit_transform([a,b])
 return ((tfidf * tfidf.T).A)[0,1]
 


# In[4]:


pfp ="F:/ME project/dataset/Topics/"
H =nx.DiGraph()
H = nx.read_edgelist(path="citation.txt", create_using =nx.DiGraph(), nodetype = int)
x = list(H.nodes())
nodes=[str(item) for item in x ]
print("done")


# In[5]:


nodes


# In[6]:


print(len(nodes))


# In[7]:


hg=[]
for node in nodes:
    if(node.startswith('30') and len(node)==6):
    #if(len(node)<=5):
        hg.append(node)


# In[8]:


print(len(hg))


# In[9]:


k=[]
for node in hg:
    node=int(node)
    k.append(node)


# In[13]:


h=[]
count=0
for node in k:
    if count<6:
        h.append(node)
        count+=1


# In[10]:


print(len(k))


# In[11]:


k


# In[12]:


candidates=[]
i=0
z=0
file=open('F:/ME project/dataset/topics.txt','a')
citu=open('F:/ME project/dataset/topiccit.txt','a')
#file.write("nodes,no.of citations,no of parent,author similarity")
for node in k:#7dig str
            j=0
            thresh = 0
            simil={}
            #int_node = int(node)
            #print(int_node)
            node_name=str(int(node))+".txt"
            fullname = pfp+node_name
            #print(fullname)
            f=open(fullname,'r')
            r=f.read()
            #if(z==0):
            file.write("\n "+str(node))
             # z=z+1
            #else:
            #  file.write("\n "+str(node))
            sus=list(H.successors(node))
            ci=len(sus)
            file.write(","+str(ci))
            i=i+1
            print(i,node)
            
            candidates=list(H.predecessors(node))
            n=len(candidates)
            
            file.write(","+str(n))
            for candidate in candidates:
                    #print(candidate)
                    cand=str(candidate)
                    candi=cand+".txt"
                    candi=pfp+candi
                    #print(candi)
                    g=open(candi,"r")
                    g=g.read()
                    #print(g)
                    #print(candi)
                    simil[candidate] = find_similar(r,g )
                    #file.write(str(simil))
                    #print(simil)
            for key in simil.keys():
              thresh = thresh+simil[key]
            #thresh=0.45
            if thresh > 0:
               thresh = thresh / len(simil.keys())
            for key in simil.keys():
                if simil[key] > thresh:
                        j=j+1
                        string = str(key)+" "+str(node)+"\n"
                        citu.write(string)
                        print(string)
                        
            file.write(","+str(j))
            
file.close()
citu.close()

