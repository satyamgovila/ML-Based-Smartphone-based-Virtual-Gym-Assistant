import argparse
import cv2 
import mediapipe as mp  #!pip install mediapipe opencv-python
import numpy as np
import evaluate
import exercises
from matplotlib import pyplot as plt
import pandas as pd
import ssl #installing temp certificates ,to resolve error in MacOS
ssl._create_default_https_context = ssl._create_unverified_context  


# Initializing mediapipe for pose detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import speech_recognition as sr # recognise speech
import playsound # to play an audio file
from gtts import gTTS # google text to speech
import random
from time import ctime # get time details
import webbrowser # open browser
import ssl
import certifi
import time
import os # to remove created audio files



r = sr.Recognizer()

def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en') 
    r = random.randint(1,200000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file) 
    playsound.playsound(audio_file) 
    print(f"Virtual Gym Assitant: {audio_string}") 
    os.remove(audio_file)
    
    
def main():
    with sr.Microphone() as source:
        try:
            speak("Hey ,Welcome to the Virtual Gym Assistant")
            speak("Mention type of exercise to be done, you have the following choices ,biceps , push ups , leg raise , squats ")
            audio = r.listen(source)
            voice_data = r.recognize_google(audio)
            speak(voice_data)
            exercise_type = voice_data
        except sr.UnknownValueError:
            speak("Sorry, I did not get that")
        except sr.RequestError:
            speak("Sorry ,the service is down")

        try:
            speak("Do you want to select random video from media dataset or start live webcam feed?")
            audio = r.listen(source)
            voice_data = r.recognize_google(audio)
            input_media_option = voice_data
            if "random" in input_media_option:
                speak("Selecting random video from media dataset")
            else:
                input_media_option = "webcam"
        except sr.UnknownValueError:
            speak("Sorry, I did not get that")
        except sr.RequestError:
            speak("Sorry ,the service is down")


        try:
            speak("Mention min detection confidence value and min tracking confidence")
            audio = r.listen(source)
            voice_data = r.recognize_google(audio)
            speak(voice_data)
            min_detection_confidence = float(voice_data)
            min_tracking_confidence = float(voice_data)
        except sr.UnknownValueError:
            speak("Sorry, I did not get that")
        except sr.RequestError:
            speak("Sorry ,the service is down")

    if exercise_type == "biceps":
        exercises.bicep_exercise(min_detection_confidence , min_tracking_confidence , "media/bicep_curl_video.mp4")
    if exercise_type == "push ups":
        exercises.push_ups_exercise(min_detection_confidence , min_tracking_confidence , "media/push_ups.mp4")
    if exercise_type == "squats":
        exercises.squates_exercise(min_detection_confidence , min_tracking_confidence , "media/Squats.mp4")
    if exercise_type == "leg raise":
        exercises.leg_raise_exercise(min_detection_confidence , min_tracking_confidence , "media/LegRaise.mp4")


        

if __name__ == "__main__":
    main()
    

  