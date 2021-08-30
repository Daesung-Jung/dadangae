# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:49:36 2021

@author: shtnr
"""
import cv2
import mediapipe
 
import datetime
import mediapipe as mp

import pandas as pd


#그리는 도구들 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
drawing_styles = mp.solutions.drawing_styles


# 필요한 칼럼 리스트
pose_tangan = ['WRIST', 'THUMB_CPC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP',
               'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP',
               'RING_FINGER_MCP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']

#변수명 생성, 위 변수명들을 x,y,z로 쪼개기
col_names=[]
for aa in pose_tangan:
    locals()[str(aa)+"_X"] = []  
    col_names.append(str(aa)+"_X")
    locals()[str(aa)+"_Y"] = []
    col_names.append(str(aa)+"_Y")
    locals()[str(aa)+"_Z"] = []
    col_names.append(str(aa)+"_Z")
    
#프레임 수 
c=0

#df에 넣을 프레임 
frame = []
col_name=[]
    
    
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
 
#불러오기 0을 넣으면 캠으로 변함
baseball_sample2="c:/data/real_pitching/samsung_s10_5g/20210806_183756"
baseball_sample = baseball_sample2.split("/")[-1]
capture = cv2.VideoCapture("%s.mp4" %baseball_sample2)


folder_path = Path("c:/data/real_pitching/pics/%s" %baseball_sample)
folder_path.mkdir(parents=True, exist_ok=True)

folder_path = Path("c:/data/real_pitching/df")
folder_path.mkdir(parents=True, exist_ok=True)


#가로 세로 이미지 사이즈,  x, y같은 경우 나중 이값을 곱해줘서 사용할 수 있다 -> 절대값
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)


#리얼타임, 현재 시간 및 날짜
now = datetime.datetime.now()
nowDatetime = now.strftime('%Y-%m-%d %H:%M:%S')


#최소 임계값 설정
with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.80, min_tracking_confidence=0.80, max_num_hands=2) as hands: 
    while (True):
        #프레임에 대한 이미지 뽑기;
        ret, image = capture.read()
        #해당 프레임 손만 추출한 결과
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not ret:
          break        
        
        if results.multi_hand_landmarks != None:
            # handLandmarks 튜플형식으로 데이터가 떨어지는 데 이때 잘 쪼갬 

            for handLandmarks in results.multi_hand_landmarks:  
                #그리기
                mp_drawing.draw_landmarks(
                    image, handLandmarks, mp_hands.HAND_CONNECTIONS),
#                    drawing_styles.get_default_hand_landmark_style(),
#                    drawing_styles.get_default_hand_connection_style())
#                cv2.imshow('MediaPipe Hands', image)
                result, encoded_img = cv2.imencode(".jpg", image)
                encoded_img.tofile('c:/data/real_pitching/pics/%s/%s.jpg' %(baseball_sample2, c))
        
                #손이 찍힌 것만 프레임 받아옴
                frame.append(c)
                for point in handsModule.HandLandmark:
                    
                    #해당하는 포인트에 대한 결과값
                    normalizedLandmark = handLandmarks.landmark[point]
                    
                    col_name = pose_tangan[point]
                    # x,y,z 을 각각 이름_x 이런식으로 데이터를 저장
                    locals()[str(col_name)+"_X"].append(normalizedLandmark.x) # * frameWidth
                    locals()[str(col_name)+"_Y"].append(normalizedLandmark.y) # * frameHeight
                    locals()[str(col_name)+"_Z"].append(normalizedLandmark.z)

                    score = results.multi_handedness[0].classification[0].score
                    label = results.multi_handedness[0].classification[0].label

        
        c+=1             
        
        #존재하는 데이터만 좌표 추출
        #프레임을 데이터로 만든 뒤 옆에 계속 쭈욱 붙임
        df=pd.DataFrame({"frame":frame,
                         "Date":nowDatetime,
                         "video_Id":baseball_sample2,
                         "score" : score, 
                         "label" : label})
        for aa in col_names:
            df_=pd.DataFrame({ str(aa) : globals()[aa]})
            df=pd.concat([df,df_], axis=1)
            
        #cv2.imshow('Test hand', frame) 
        if cv2.waitKey(1) == 27:
            break
        
df.to_csv("c:/data/real_pitching/df/%s.csv" %baseball_sample2)        
        
