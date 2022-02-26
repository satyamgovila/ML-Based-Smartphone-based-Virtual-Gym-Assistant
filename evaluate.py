import numpy as np
import cv2 
import mediapipe as mp  #!pip install mediapipe opencv-python
import numpy as np
import evaluate
from matplotlib import pyplot as plt
import pandas as pd
import ssl #installing temp certificates ,to resolve error in MacOS
ssl._create_default_https_context = ssl._create_unverified_context  

# Initializing mediapipe for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose



def calculate_angle(x1,x2,x3):
    x1 = np.array(x1) # Initial Point
    x2 = np.array(x2) # Mid Point
    x3 = np.array(x3) # Terminal Point


    angle_radians = np.arctan2(x3[1]-x2[1], x3[0]-x2[0]) - np.arctan2(x1[1]-x2[1], x1[0]-x2[0])

    angle_degree = np.abs(angle_radians*180.0/np.pi)

    if angle_degree > 180.0:
        angle_degree = 360-angle_degree

    return angle_degree 



def display_status(image , counter , stage , results):
    # Render curl counter
    # Setup status box
    cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

    # Rep data
    cv2.putText(image, 'REPS:', (15,12), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(counter), 
            (10,60), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    # Stage data
    cv2.putText(image, 'STAGE:', (65,12), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, stage, 
            (60,60), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)


    # Render detections
    mp_drawing.draw_landmarks(
                        image, 
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                         )    
    

