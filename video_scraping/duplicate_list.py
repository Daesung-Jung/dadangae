# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 08:34:35 2020

@author: shtnr
"""


#####################################################################################################   
#                                                                                                   #
#                                                                                                   #   
#           용량이 작거나 빠진 리스트 확인                                                               #
#                                                                                                   #
#                                                                                                   #
#####################################################################################################


#1. 용량이 작은 파일 골라내기

import glob
import pandas as pd
from pathlib import Path
import os
path_dir ="X:/Dadangae/rawdata_video_broadcasting/" 
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
        dt.append(Path((qq+bb)).stat().st_size)



our_df=pd.DataFrame({"cate":cate,
                 "tt":tt,
                 "dt":dt})


#파일 받아오기
file = "C:/code/baseball_pitchdesign/video_scraping/data/full_videos_list.csv"
df=pd.read_csv(file)

#컬럼명 지정
df.columns=['a','team','site']

#team컬럼에 [다시보기]제거
df['team']=df['team'].str.replace('\[다시보기\] ','')

#필요없는 컬럼 제거
df.drop(['a'],axis=1,inplace=True)

#dh(더블헤더)컬럼 생성
def read_dh(s):
  if 'DH1' in s:
    return '1'
  elif 'DH2' in s:
    return '2'
  return '0' 

df['dh']=df['team'].apply(lambda x:read_dh(x))

#team컬럼 형식 통일
df['team']=df['team'].str.replace('월_','월 ')
df['team']=df['team'].str.replace('일 ','월 ')

#team컬럼 '_'기준으로 나눠서 날짜, 파일번호, 팀 컬럼 생성
df['date']=df['team'].str.split('_').str[1]
df['number']=df['team'].str.split('_').str[2]
df['team']=df['team'].str.split('_').str[0]

#date컬럼 형식 통일
df['date']=df['date'].str.replace(' (DH1)','')

#number컬럼 형식 통일
df['number']=df['number'].str.replace('부','')

#team컬럼 형식을 예) 롯데 vs 삼성 으로 되있는걸 LTSS 이런 형식으로 통일
df['team']=df['team'].str.replace(' vs ','')
df['team']=df['team'].str.replace('롯데','LT')
df['team']=df['team'].str.replace('KIA','HT')
df['team']=df['team'].str.replace('키움','WO')
df['team']=df['team'].str.replace('두산','OB')
df['team']=df['team'].str.replace('SSG','SK')
df['team']=df['team'].str.replace('삼성','SS')
df['team']=df['team'].str.replace('한화','HH')

#team컬럼 필요없는 부분 제거
df['team']=df['team'].str[:4]

#날짜 형식 통일
df['mon']=df['date'].str.split('월 ').str[0]
df['day']=df['date'].str.split('월 ').str[1]
df['day']=df['day'].str.replace('일','')

def change_date(s):
    if len(s)==1:
        return str('0'+s)
    else:
        return str(s)
    
df['mon']=df['mon'].apply(lambda x:change_date(x))
df['day']=df['day'].apply(lambda x:change_date(x))
df['date']='2021'+df['mon']+df['day']
df.drop(['mon','day'],axis=1,inplace=True)


#전체 우리가 가지고 있는 리스트
df['gameId'] = df['date'] + df['team'] + df['dh'] + "2021-" + df['number']


need = df['gameId']
need = pd.DataFrame(need.unique())[0]

our_=our_df['tt'].str.split(".mp4").str[0]


#우리가 다운로드 받지 못했던 추가 리스트 

'''
4      20210321KTOB02021-1
470    20210423SKWO02021-1
631    20210502WONC02021-1
688    20210508NCKT02021-5
'''
need[~need.isin(our_)]




file= "C:/code/baseball_pitchdesign/video_scraping/data/need_list_2021.csv"
need_list=pd.read_csv(file)

df['gameId']

#진짜 필요한 리스트
qq=df[(df['gameId']+".mp4").isin(need_list[need_list['complete']=="x"]['tt'])]
qq.to_csv("C:/code/baseball_pitchdesign/video_scraping/data/final_need_2021.csv")

