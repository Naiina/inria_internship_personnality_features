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
from pydub import AudioSegment
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score


direct_mp4 = "../chalearn_data/training_set1/training80_01_mp4/"
direct_wav = "../chalearn_data/training_set1/training80_01_wav/"
l_file_names_mp4 = os.listdir(direct_mp4)
csv_file = "../chalearn_data/FI_pairwise_data.csv"
df_pairewise_data = pd.read_csv(csv_file)  



'''
Convert mp4 files into wav
'''

# =============================================================================

# def convert_files(l_file_names_mp4,direct_mp4,direct_wav):
#     for file_name_mp4 in l_file_names_mp4:
#         file_name_wav = file_name_mp4[:-4]+".wav"
#         track = AudioSegment.from_file(direct_mp4+file_name_mp4,  format='m4a')
#         file_handle = track.export(direct_wav+file_name_wav, format='wav')
# convert_files(l_file_names_mp4,direct_mp4,direct_wav)
# 
# =============================================================================



def find_first_occurence_row_csv(csv_file,file_name_mp4):

    with open(csv_file, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if file_name_mp4 in row:
                return row
            
def video_position(row,video_name):
    if video_name == row[0]:
        return "LEFT"
    if video_name == row[1]:
        return "RIGHT"
    else:
        return "Not in this row"
    
def personality_trait_in_row(row,video_position):
    """
    Parameters
    ----------
    row : list of str
        DESCRIPTION.
    video_position : str
        "LEFT" or "RIGHT"

    Returns
    -------
    list of str
        list of the "personality_trait" labeled as "video_position" 

    """
    
    l_personality_traits_of_video = [] 
    l_personality_traits = ['friendly', 'authentic', 'organized', 'comfortable', 'imaginative']
    for i in range(5):
        if video_position == row[i+2]:
            elem = l_personality_traits[i]
            l_personality_traits_of_video.append(elem)
    return l_personality_traits_of_video

def compute_score_video(csv_file,file_name_mp4):
    nb_of_occurences = 0
    d_big_fives = {'friendly':0, 'authentic':0, 'organized':0, 'comfortable':0, 'imaginative':0}
    
    with open(csv_file, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        for row in csv_reader:
            
            if file_name_mp4 in row:
                nb_of_occurences +=1
                video_pos = video_position(row,file_name_mp4)
                l_perso_traits = personality_trait_in_row(row,video_pos)
                for elem in l_perso_traits:
                    d_big_fives[elem]+=1
                    
    # compute the average of each trait
    if nb_of_occurences !=0:
        for trait in d_big_fives:
            d_big_fives[trait] = d_big_fives[trait]/nb_of_occurences
    return d_big_fives


def create_data_set(l_file_names_mp4,direct_wav):
    X = []
    to_exclude = []
    for file_name_mp4 in l_file_names_mp4:
        file_name_wav = file_name_mp4[:-4]+".wav"
        samplerate,data = wav.read(direct_wav+file_name_wav)
        if np.shape(data) == (674816, 2):
            data0 = data[:,0]
            X.append(data0) 
        else:
            to_exclude.append(file_name_mp4)
        #y.append(list(df_big_five.loc[file_name_mp4,:]))
    print("we keept only the data with shape (674816, 2). TO DO: what to do with the other videos?")
    return X,to_exclude


'''
compute mfcc and mel. take not the right data. To change
'''


# =============================================================================
# l_mfcc = []
# l_mel = []
# l_features = []
# l_data = []
# for file_name_mp4 in l_file_names_mp4[:4]:
#     file_name_wav = file_name_mp4[:-4]+".wav"
#     samplerate,data = wav.read(direct_wav+file_name_wav)
#     #data = data[:,0]
#     l_data.append(data)
#     l_mfcc.append(mfcc(data,samplerate,nfft=512*3)) # by default 13 features
#     l_mel.append(logfbank(data,samplerate,nfft=512*3)) #mel filter bank, default nb featuers: 26
#     mem = find_first_occurence_row_csv(csv_file,file_name_mp4)    
#     l_features.append(mem)
# 
# df_features = pd.DataFrame(l_features) 
# df_features.columns =['videoLeft','videoRight','friendly','authentic','organized','comfortable','imaginative','interview']
# 
# =============================================================================


"""
Create csv file of features of each video using "FI_pairwise_data.csv"

"""


l_dict = []

for file_name_mp4 in l_file_names_mp4:
    d_big_fives = compute_score_video(csv_file,file_name_mp4)
    l_big_fives = list(d_big_fives.values())
    l_dict.append(l_big_fives)
    

df_big_five = pd.DataFrame(l_dict, columns=['friendly','authentic','organized','comfortable','imaginative'], index=l_file_names_mp4)
#df_big_five.to_csv("../chalearn_data/training_set1_training80_01_mp4.csv")



"""
Create training and testing sets

"""


X,to_exclude = create_data_set(l_file_names_mp4,direct_wav)
np_X = np.stack( X, axis=0 )  #all data are not in stereo. For now I just took one of the data columns of stereo videos

for elem in to_exclude:
    df_big_five = df_big_five.drop(elem, axis=0)
    
np_y = df_big_five.to_numpy()



def first_classifier_BC(method,np_X,np_y,trait,test_set_ratio=0.25): 
    if trait == 'friendly':
        Y = np_y[:,0]
    if trait == 'authentic':
        Y = np_y[:,1]
    if trait == 'organized':
        Y = np_y[:,2]
    if trait == 'comfortable':
        Y = np_y[:,3]
    if trait == 'imaginative':
        Y = np_y[:,4]
    
    X_train, X_test, y_train, y_test = train_test_split( np_X, Y, test_size=test_set_ratio, random_state=42)
    
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



first_classifier_BC('log_reg',np_X,np_y,'imaginative')









