# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:03:43 2023

@author: Nina Gregorio
"""

from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("file.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])