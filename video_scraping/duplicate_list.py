# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 08:34:35 2020

@author: shtnr
"""


import glob
import pandas as pd

import os
path_dir ="Z:/rawdata_video_broadcasting/" 
file_list = os.listdir(path_dir)

cate=[]
tt=[]

for aa in range(0,len(file_list)-1):
    qq = path_dir+file_list[aa]+"/"
    for bb in os.listdir(qq):
        cate.append(aa+1)
        tt.append(bb) 

len(cate)
len(tt)


df=pd.DataFrame({"cate":cate,
                 "tt":tt})

df_=df[df['tt'].isin(df[df['tt'].duplicated()]['tt'])]

df_=df_.sort_values(['tt'])
