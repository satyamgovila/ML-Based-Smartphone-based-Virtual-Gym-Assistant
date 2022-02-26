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


def bicep_exercise(min_detection_confidence , min_tracking_confidence ,media ):
    # cap = cv2.VideoCapture(1)
    if media == "webcam":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(media)

    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Calculate angle
                angle = evaluate.calculate_angle(shoulder, elbow, wrist)

                # Visualize angle
                cv2.putText(image, str(angle), 
                               tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )

                # Curl counter logic
                if angle > 130:
                    stage = "down"
                if angle < 60 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)
            except:
                pass
            
            evaluate.display_status(image , counter , stage , results)           

            cv2.imshow('Live Virtual Assistant', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        
        
def push_ups_exercise(min_detection_confidence , min_tracking_confidence ,media ):
        # cap = cv2.VideoCapture(1)
    if media == "webcam":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(media)



    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle
                angle = evaluate.calculate_angle(shoulder, elbow, wrist)
         
                cv2.putText(image, "Angle between Joints:"+ str(angle), tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)



                # Curl counter logic
                if angle > 130:
                    stage = "up"
                if angle < 90 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)

            except:
                pass
            
            evaluate.display_status(image , counter , stage , results)

            cv2.imshow('Live Virtual Assistant', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        
        
        
        
def leg_raise_exercise(min_detection_confidence , min_tracking_confidence ,media ):

    # cap = cv2.VideoCapture(1)
    if media == "webcam":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(media)



    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                # Calculate angle
                angle = evaluate.calculate_angle(knee, hip, shoulder)         
                cv2.putText(image, "Angle between Joints:"+ str(angle), tuple(np.multiply(shoulder, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)



                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 95 and stage =='down':
                    stage="up"
                    counter +=1
                    print(counter)



            except:
                pass
            evaluate.display_status(image , counter , stage , results)

            cv2.imshow('Live Virtual Assistant', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        
        
        
def squates_exercise(min_detection_confidence , min_tracking_confidence ,media ):
    if media == "webcam":
        cap = cv2.VideoCapture(1)
    else:
        cap = cv2.VideoCapture(media)



    # Curl counter variables
    counter = 0 
    stage = None

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence = min_detection_confidence, min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate angle
                angle = evaluate.calculate_angle(hip, knee, ankle)        
                cv2.putText(image, "Angle between Joints:"+ str(angle), tuple(np.multiply(hip, [640, 480]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                


                # Curl counter logic
                if angle > 120:
                    stage = "up"
                if angle < 90 and stage =='up':
                    stage="down"
                    counter +=1
                    print(counter)



            except:
                pass
            
            evaluate.display_status(image , counter , stage , results)

            cv2.imshow('Live Virtual Assistant', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
        

