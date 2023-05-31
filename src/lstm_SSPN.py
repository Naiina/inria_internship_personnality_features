# -*- coding: utf-8 -*-
"""
Created on Wed May 10 16:24:46 2023

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 11:42:31 2023

@author: Lenovo
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os


import pandas as pd
import datetime
import time
from csv import reader
import matplotlib.pyplot as plt
from pydub import AudioSegment
import wave
from scipy.io import wavfile
from scipy.io.wavfile import write
import numpy as np
import os
import contextlib
import torchaudio
import wave

def get_max_size_data(audio_path):
    l_direct= os.listdir(audio_path)
    frames = []
    for direct in l_direct:
        files = os.listdir(audio_path +direct)
        for elem in files:
            with contextlib.closing(wave.open(audio_path+direct+'/'+elem,'r')) as f:
                frames.append( f.getnframes())
    return max(frames)
            
def is_positive(n):
    if n>0:
        return 1
    else:
        return 0


class UrbanSoundDataset(Dataset):
    # Wrapper for the UrbanSound8K dataset
    # Argument List
    # path to the UrbanSound8K csv file
    # path to the UrbanSound8K audio files
    # list of folders to use in the dataset

    
    def __init__(self, csv_path, path_audio, max_file_len,folderList): #folderList):
        csvData = pd.read_csv(csv_path)
        # initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        # loop through the csv entries and only add entries from folders in the folder list
        for i in range(0, len(csvData)):
            
            if csvData.iloc[i, 8] in folderList:
                self.file_names.append(csvData.iloc[i, 2])
                #lab = torch.tensor([csvData.iloc[i, j] for j in range(2,7)])
                self.labels.append(csvData.iloc[i, 9])
                self.folders.append(csvData.iloc[i, 8])
            
            #self.file_names.append(csvData.iloc[i, 0])
            #self.labels.append(is_positive(csvData.iloc[i, 1]))
            
        self.path_audio = path_audio
        self.max = max_file_len
        self.folderList = folderList
        
        

    def __getitem__(self, index):
        # format the file path and load the file
        max_frames = self.max
        path = self.path_audio + str(self.folders[index]) + "/" +self.file_names[index]+".wav"
        soundData, sample_rate = torchaudio.load(path)
        #soundData = torch.mean(sound, dim=0, keepdim=True)
        tempData = torch.zeros([1, max_frames])  # tempData accounts for audio clips that are too short
        
        
        if soundData.numel() < max_frames:
            tempData[:, :soundData.numel()] = soundData
        else:
            tempData = soundData[:, :max_frames]

        soundData = tempData

        #mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(soundData)  # (channel, n_mels, time)
        #mel_specgram_norm = (mel_specgram - mel_specgram.mean()) / mel_specgram.std()
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate)(soundData)  # (channel, n_mfcc, time)
        mfcc_norm = (mfcc - mfcc.mean()) / mfcc.std()
        # spectogram = torchaudio.transforms.Spectrogram(sample_rate=sample_rate)(soundData)
        #feature = torch.cat([mel_specgram, mfcc], axis=1)
        feature = mfcc
        #print("f",len(feature[0].permute(1, 0)))
        return feature[0].permute(1, 0), self.labels[index]

    def __len__(self):
        return len(self.file_names)
    
    





class AudioLSTM(nn.Module):

    def __init__(self, n_feature=5, out_feature=5, n_hidden=128, n_layers=2, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_feature = n_feature

        self.lstm = nn.LSTM(self.n_feature, self.n_hidden, self.n_layers, dropout=self.drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, out_feature)

    def forward(self, x, hidden):
        # x.shape (batch, seq_len, n_features)
        l_out, l_hidden = self.lstm(x, hidden)

        # out.shape (batch, seq_len, n_hidden*direction)
        out = self.dropout(l_out)

        # out.shape (batch, out_feature)
        out = self.fc(out[:, -1, :])

        # return the final output and the hidden state
        return out, l_hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden
    
        
def train(model, epoch):
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        #print(target.size())
        #print(data.size())
        target = target.to(device)

        model.zero_grad()
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        #print(output.size())
        loss = criterion(output, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        #if batch_idx % log_interval == 0: #print training stats
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss))
    return loss
            
            
def test(model, epoch):
    model.eval()
    correct = 0
    y_pred, y_target = [], []
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        
        output, hidden_state = model(data, model.init_hidden(hyperparameters["batch_size"]))
        
        pred = torch.max(output, dim=1).indices
        correct += pred.eq(target).cpu().sum().item()
        y_pred = y_pred + pred.tolist()
        y_target = y_target + target.tolist()
    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct
    
hyperparameters = {"lr": 0.005, "weight_decay": 0.0001, "batch_size": 100, "in_feature": 40, "out_feature": 5}

device = torch.device("cpu")
print(device)

#csv_path = '/kaggle/input/urbansound8k/UrbanSound8K.csv'
#file_path = '/kaggle/input/urbansound8k/'

path_audio = "../SSPNet-Speaker-Personality-Corpus/Audio_clips/"
csv_path = "../SSPNet-Speaker-Personality-Corpus/Personality_Scores/Score_001.csv"
#l_folders = [elem for elem in os.listdir(path_cut_before) if os.path.isdir(path_cut_before+elem)]
max_file_len = get_max_size_data(path_audio)


train_set = UrbanSoundDataset(csv_path, path_audio, max_file_len,["training_set"])
test_set = UrbanSoundDataset(csv_path, path_audio, max_file_len,["test_set"])

#print("Train set size: " + str(len(train_set)))
#print("Test set size: " + str(len(test_set)))




kwargs = {}  # needed for using datasets on gpu

train_loader = torch.utils.data.DataLoader(train_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=hyperparameters["batch_size"], shuffle=True, drop_last=True, **kwargs)

model = AudioLSTM(n_feature=hyperparameters["in_feature"], out_feature=hyperparameters["out_feature"])
model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
criterion = nn.CrossEntropyLoss()
clip = 5  # gradient clipping

log_interval = 10
l_loss = []
l_correct = []
l_epoch = []
test_size = len(test_loader.dataset)
for epoch in range(1, 31):
    # scheduler.step()
    loss = train(model, epoch)
    correct = test(model, epoch)/test_size
    
    l_loss.append(loss)
    l_correct.append(correct)
    l_epoch.append(epoch)
    
    


l = [l_loss[i].data.item() for i in range(len(l_loss))]
d = {"epoch":l_epoch,"loss":l,"correct":l_correct}
df = pd.DataFrame.from_dict(d)
df.to_csv("perf_128_lr_0_005_drop_0_01.csv")
    
