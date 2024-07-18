[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/5mCoF9-h)
# TOBB ETÜ ELE495 - Capstone Project

# Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Video Demo](#usage)
- [Screenshots](#screenshots)
- [Acknowledgements](#acknowledgements)

## Introduction
Provide a brief overview of the project, its purpose, and what problem it aims to solve.
The aim of this project is to receive a license plate number sent by the mobile application by Jetbot via an API and park it autonomously in the license plate area on the platform. While carrying out the project, software such as image processing, artificial intelligence and mobile development were used.

![S8374b6d2789f4a61bca6abbf57b5ea3dK jpg_640x640Q90 jpg_](https://github.com/user-attachments/assets/af9d4acc-5c85-47eb-8595-96dfc061a3b9)


## Features
List the key features and functionalities of the project.
- Hardware: The hardware components used (should be listed with links)
  - Jetbot AI Kit (https://www.waveshare.com/product/jetbot-ai-kit.htm) 
- Applications
  - Operating System and packages
  - Python 3.x
  - JetBot SDK
  - OpenCV
  - Flask
  - Torch
- Services
  -Android Mobile App

## Installation
!! You must have minimum nvidia jetson nano jetpack:jp41 version for installation !!

Commands you need to execute to start the python application.
you need to execute this commands in the min jp41 jetbot docker container
```bash
#  commands
git clone https://github.com/ELE495-2324Summer/capstoneproject-tripod-grup-6/tree/main
cd capstoneproject-tripod-grup-6
cd code
python3 main.py
```
after this bash commands flask server will be running on the jetbot with jetbot's ip

For sending plate number to flask server we will use an Android mobile application. You can use APK directly or you can modify the app code and build on Android Studio IDE.


## Usage
Run the python script from jetbot and send parking command with number plate from mobile app.
![WhatsApp Image 2024-07-16 at 03 44 06](https://github.com/user-attachments/assets/7a4164fd-1abc-4eaf-ac27-87316e012eb2)
![WhatsApp Image 2024-07-16 at 03 44 05](https://github.com/user-attachments/assets/39982023-6cb7-459f-9d17-6e9c565d4ab5)


## Video Demo
you can find a video that shows jetbot parking to number 4 via mobile application command.
https://www.youtube.com/watch?v=vKzyPDBvfdU


## Acknowledgements
Give credit to those who have contributed to the project or provided inspiration. Include links to any resources or tools used in the project.

[Contributor 1](https://github.com/user1)
[Resource or Tool](https://www.nvidia.com)
