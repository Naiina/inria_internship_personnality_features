# -*- coding: utf-8 -*-
"""
Created on Thu Mar 6 10:03:43 2023

@author: Nina Gregorio
"""

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import pandas as pd
from csv import reader
import os


#wav_file = "chalearn_data_first_week/wav/ztmNiy8D3c.wav"
#samplerate,data = wav.read(wav_file)
#data = data[:,0]
#mfcc_feat = mfcc(data,samplerate,nfft=512*3) # by default 13 features
#fbank_feat = logfbank(data,samplerate,nfft=512*3) #mel filter bank, default nb featuers: 26

csv_file = "chalearn_data_first_week/FI_pairwise_data.csv"

df_pairewise_data = pd.read_csv(csv_file)  


def find_first_occurence_row_csv(csv_file,wav_file):

    with open(csv_file, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        mp4_file = wav_file[:-4]+".mp4"
        for row in csv_reader:
            if mp4_file in row:
                return row

            
l_file_names = os.listdir("chalearn_data_first_week/wav")
l_mfcc = []
l_mel = []
l_features = []
for elem_wav in l_file_names:
    samplerate,data = wav.read("chalearn_data_first_week/wav/"+elem_wav)
    data = data[:,0]
    l_mfcc.append(mfcc(data,samplerate,nfft=512*3)) # by default 13 features
    l_mel.append(logfbank(data,samplerate,nfft=512*3)) #mel filter bank, default nb featuers: 26
    mem = find_first_occurence_row_csv(csv_file,elem_wav)
    
    l_features.append(mem)

df_features = pd.DataFrame(l_features) 
df_features.columns =['videoLeft','videoRight','friendly','authentic','organized','comfortable','imaginative','interview']

def first_classifier_BC(method,wav_file_A1,wav_file_A2,csv_file,test_set_ratio=0.25):

    l_data_f,l_label_f = generate_X_Y_feedback(wav_file_A1,wav_file_A2,csv_file)
    l_data_r,l_label_r = generate_X_Y_resp(wav_file_A1,wav_file_A2,csv_file)
    
    l_data_set = l_data_f + l_data_r
    l_label_set = l_label_f + l_label_r
    
    X = np.array(l_data_set)
    y = np.array(l_label_set)
   
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_set_ratio, random_state=42)
    
    if method == 'log_reg':
        classifier = LogisticRegression()
    if method == 'Neigh':
        classifier = KNeighborsClassifier()
    if method == 'Random_forest':
        classifier = RandomForestClassifier()
    

    classifier.fit(X_train, y_train)
    y_predict = []
    for i in range(len(y_test)):
        y_predict.append( classifier.predict([X_test[i]]))
    print(balanced_accuracy_score(y_test, y_predict))
    print(cohen_kappa_score(y_test, y_predict))
    return X,y