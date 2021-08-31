# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:35:23 2021

@author: shtnr
"""


import glob
import pandas as pd
from pathlib import Path
import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




path_dir ="c:/data/edited_video/" 
file_list = os.listdir(path_dir)

cate=[]
tt=[]
dt=[]

#폴더명, 파일명, 용량 
for aa in range(0,len(file_list)):
    print(aa)
    qq = path_dir+file_list[aa]+"/"
    for bb in os.listdir(qq):
        cate.append(aa+1)
        tt.append(bb) 
#        dt.append(Path((qq+bb)).stat().st_size)

len(tt)


import pandas as pd



df1=pd.read_csv("C:/data/conf_by_frame/20210415LGWO02021-1.csv")



def conf_by_frame(data_,threshold):
    data_['conf_']=data_['conf'].str.split("tensor\(").str[1].str.split(",").str[0]
    data_['conf_']=data_['conf_'].astype(float)
    data_['frame']=data_['frame'].astype(int)
    data__= data_[data_.conf_>threshold]

    fig, ax = plt.subplots()
    ax.plot(data__['frame'], data__['conf_'])


conf_by_frame(df1, 0.7)



###############################
conf_frame_1=pd.read_csv("C:/data/conf_by_frame/20210415LGWO02021-1.csv")
conf_frame_2=pd.read_csv("C:/data/conf_by_frame/20210415NCSK02021-4.csv")

conf_frame_1['conf_']=conf_frame_1['conf'].str.split("tensor\(").str[1].str.split(",").str[0]
conf_frame_1['conf_']=conf_frame_1['conf_'].astype(float)
conf_frame_1['frame']=conf_frame_1['frame'].astype(int)



conf_frame_2['conf_']=conf_frame_2['conf'].str.split("tensor\(").str[1].str.split(",").str[0]
conf_frame_2['conf_']=conf_frame_2['conf_'].astype(float)
conf_frame_2['frame']=conf_frame_2['frame'].astype(int)



conf_frame_1_ = conf_frame_1[conf_frame_1.conf_>0.8]
conf_frame_2_ = conf_frame_2[conf_frame_2.conf_>0.8]


plt.plot(conf_frame_1['frame'],conf_frame_1['conf_'])



fig, ax = plt.subplots()
ax.plot(conf_frame_1_['frame'], conf_frame_1_['conf_'])


fig, ax = plt.subplots()
ax.plot(conf_frame_2_['frame'], conf_frame_2_['conf_'])





