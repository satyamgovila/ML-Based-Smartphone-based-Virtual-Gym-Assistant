# Smartphone-Based-Gym-Assistant
This repository contains all the documentation, code, thesis , and media filess required for the evaluation of project work.

The application is deployed on AWS EC2 instance and uses streamlit for integrating web-interface for user interactivity.


**Steps for deploying virtual gym assistant application on local:-

1. Setup AWS EC2 on local system using ssh command

    AWS EC2 instance used- Ubuntu Server 14.04 LTS (Free tier version)

    ssh -i "streamlit.pem" ubuntu@ec2-3-14-151-6.us-east-2.compute.amazonaws.com
    git clone <repo link>
  
    cd <directory>
  
2. Running Streamlit web application
  
    streamlit run app_deploy.py
  
 
**Command used to run application CLI mode:-
  
    python app_cli.py --min_detection_confidence 0.5 --min_tracking_confidence 0.5 --exercise_type biceps --media media/bicep_curl_video.mp4
    python app_cli.py --min_detection_confidence 0.5 --min_tracking_confidence 0.5 --exercise_type push_ups --media media/push_ups.mp4
    python app_cli.py --min_detection_confidence 0.5 --min_tracking_confidence 0.5 --exercise_type squats --media media/squats.mp4
    python app_cli.py --min_detection_confidence 0.5 --min_tracking_confidence 0.5 --exercise_type leg_raise --media media/LegRaise.mp4

    python app_cli.py --min_detection_confidence 0.5 --min_tracking_confidence 0.5 --exercise_type biceps --media webcam
  

**Command used to run application for voice recognition functionality
  
  python app_voice.py
  
  
