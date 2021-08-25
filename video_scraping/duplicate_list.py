# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 08:34:35 2020

@author: shtnr
"""


import glob
import pandas as pd
from pathlib import Path
import os
path_dir ="Z:/rawdata_video_broadcasting/" 
file_list = os.listdir(path_dir)

cate=[]
tt=[]
dt=[]

#폴더명, 파일명, 용량 
for aa in range(0,len(file_list)-1):
    qq = path_dir+file_list[aa]+"/"
    for bb in os.listdir(qq):
        cate.append(aa+1)
        tt.append(bb) 
        dt.append(Path((qq+bb)).stat().st_size)



df=pd.DataFrame({"cate":cate,
                 "tt":tt,
                 "dt":dt})

#분할 해서 변수명 따로 만들기
df['game_id']=df['tt'].str.split("-").str[0]
df['series']=df['tt'].str.split("-").str[1].str.split(".").str[0]


#마지막이 아닐 때 2.6기가가 아닌 것 
game_id_max=pd.DataFrame(df.groupby('game_id').series.agg('max').reset_index())
game_id_max['tt'] = game_id_max['game_id']+'-'+ game_id_max['series']+'.mp4'


#마지막 행 제외 데이터 
not_last_data = df[~df['tt'].isin(game_id_max['tt'])]
#메가바이트 단위 전환
not_last_data['dt'] = not_last_data['dt'].astype(float)/(1024*1024*1024)
not_last_data['dt'].plot()
not_last_data['dt'].hist()

#툭 튀는 최대값 확인하기 -> 3.1기가? 왜지 일단 보류 
not_last_data[not_last_data['dt'].max()==not_last_data['dt']]



not_last_data[not_last_data['dt']<2.57]
