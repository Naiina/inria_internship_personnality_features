# -*- coding: utf-8 -*-
"""
Created on Wed May 31 11:51:19 2023

@author: Lenovo
"""
import pandas as pd

csv_path = "../SSPNet-Speaker-Personality-Corpus/Personality_Scores/Score_001.csv"

df = pd.read_csv(csv_path)
#print(df.loc[0,["Extraversion","Agreeableness","Conscientiousness","Neuroticism","Openness"]])
r = []

l = []

for i in range(len(df)):
    if "8" in df.iloc[i,0]:
        l.append("test_set")
    else:
        l.append("training_set")
        
df1 = pd.DataFrame(l,columns=['folder'])
df2 = pd.concat([df, df1],axis = 1)
df2.to_csv(csv_path)

df = pd.read_csv(csv_path)
        
for i in range(len(df)):
    dfl=df.loc[i,["Extraversion","Agreeableness","Conscientiousness","Neuroticism","Openness"]]
    l = dfl.tolist()
    r.append(l.index(max(l)))

df1 = pd.DataFrame(r,columns=['label'])
df2 = pd.concat([df, df1],axis = 1)
df2.to_csv(csv_path)
    


        
