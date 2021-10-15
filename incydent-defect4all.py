# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 11:08:17 2021

@author: M017824
"""

#dobre dziala

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer

df_incidents1 = pd.read_excel('CDW_incidents.xlsx')

take_rows=len(df_incidents1)
df_incidents=df_incidents1.iloc[0:take_rows,:] #take only ...

cc=df_incidents['Description'][:]
dd1=list(cc)

number_of_incid=len(dd1)
sentences = dd1

model = SentenceTransformer('/codebert-base')
incidents_embeddings = model.encode(sentences)

embeddings1=incidents_embeddings

df_defects1 = pd.read_excel('CDW defects2.xlsx')

take_defects=len(df_defects1)

#take_rows=999 #take defects
take_rows=take_defects
df_defects=df_defects1.iloc[0:take_rows,:] #take only ...

cc=df_defects['Description'][:]
dd=list(cc)

number_of_defects=len(dd)
sentences = dd


defects_embeddings = model.encode(sentences)

embeddings2=defects_embeddings


show_similar=5

#for i in range(number_of_cases): 
for i in range(number_of_incid): 
    print(i)
    euc_dis=euclidean_distances(
        [embeddings1[i]],
        embeddings2[0:]    
        )
    euc_dis=euc_dis.flatten()
    sort_ind=np.argsort(euc_dis)
    euc_dis_sort=sorted(euc_dis)

    sort_indl=sort_ind.flatten()
    zz1=df_defects['Description'][sort_indl]
    zz0=zz1[0:show_similar]
    incident=dd1[i] #put incident at top
    zz2=pd.concat([pd.Series([incident]),zz0])  
    
    zz=zz2
    if i==0:
        similar=np.array(list(zz))
        similar.reshape(-1,1)
#  adds first column
    else:
        zz1=np.array(list(zz)).reshape(-1,1)
        similar=np.column_stack((similar,zz1)) #here are descriptions sorted



