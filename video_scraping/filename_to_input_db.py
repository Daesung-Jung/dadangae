# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 10:13:33 2021

@author: shtnr
"""

#1. 용량이 작은 파일 골라내기

import glob
import pandas as pd
from pathlib import Path
import os
from datetime import timedelta
from datetime import timezone
import numpy as np

#불러올 위치
#저장 위치

def file_name_list(
        drive_list="Y:/Dadangae/Drive_Google/zoom_in/labels",
        down_dir="c:/data/filename_list.csv"
        ):
    
        
    gameId= []
    no= []
    frame= []
    conf= []
    time= []
    FPS = 60.0
    #최하위 위치와 그 파일명 
    for (path, directory, files) in os.walk(str(drive_list)):
        print(path)
        if path==drive_list:
            continue
        
        #하위 위치에서 폴더명이 곧 gamdId
        #그 위치에서 해당 파일들의 폴더 no 
         
        gameId.append(path.split("\\")[-2])
        no.append(path.split("\\")[-1])
        
        if len(files)!=0:        
            conf_=float(files[0].split("(")[1].split(",")[0])
            conf.append(conf_)
            frame_=int(files[0].split("_")[0])
            frame.append(frame_)
            
            time_= timedelta(seconds=int(frame_/FPS))
            time.append(timezone(time_))
            
        else:
            conf.append(np.nan)
            frame.append(np.nan)
            time.append(np.nan)
            
    
    df__=pd.DataFrame({"conf":conf,
                     "frame":frame,
                             "gameId":gameId,
                             "no":no,
                             'time':time})
    
    
    
    
    #이상한 거 제거
    df__=df__[~df__.gameId.isin([str(drive_list),'1','2','3'])]
    df__=df__[~df__['no'].isin(['0( 0~1은 붙은거)','.ipynb_checkpoints'])]
    
    
    
    df__['no']=df__['no'].astype(int)
    df__ = df__.sort_values(['gameId','frame'])     
    
    df__['투수'] = np.nan
    df__['이닝'] = np.nan
    df__['초말'] = np.nan
    df__['투구수'] = np.nan
    df__['드롭_비디오'] = np.nan
    
    df__.to_csv(str(down_dir),encoding='cp949')
    
    


file_name_list()    

