# -*- coding: utf-8 -*-
"""
Created on Wed May 31 12:04:28 2023

@author: Lenovo
"""
import pandas as pd
import matplotlib.pyplot as plt

n_hidden = "256"
lr = "0_005"
drop = "0_1"
df = pd.read_csv("perf_"+n_hidden+"_lr_0_005.csv")


l_loss = list(df["loss"])
l_accuracy =  list(df["correct"])

plt.plot(l_loss, label = "loss")
plt.plot(l_accuracy, label = "accuracy (%)")
plt.xlabel('nb epochs')
plt.legend()
plt.hlines(0.5,0,40, colors='k', linestyles='dashed')
plt.title("lstm 2 layers, "+n_hidden+" hidden, lr = "+lr+", drop = "+drop)
plt.savefig("lstm 2 layers, "+n_hidden+" hidden"+lr+", drop = "+drop)
plt.show()

