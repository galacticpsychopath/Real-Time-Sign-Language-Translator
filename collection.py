import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt
#Define the actions (signs) you want to detect (e.g., 'hello', 'thanks', 'iloveyou').
actions = ["hello", "thanks", "iloveyou"]
num_sequences = 30
num_frames = 30 
results=holistic.process(image)#This contains all the body landmarks.
                               #Each landmark has x, y, z (and sometimes visibility).
# Loop through each action and collect specific number of sequences (videos) and frames per sequence.
# Extracted keypoints (Face, Pose, Left Hand, Right Hand) are flattened and saved to a folder named "MP_Data"
for action in actions : 
    for sequence in range(1, num_sequences+1):
        try:    
            os.makedirs(os.path.join("MP_Data", action, str(sequence)))
        except: 
            pass 
        #we will have 30 sequences for each action and 30 frames for each sequence so the ai have verity of every action

def extract_keypoints(results):
    pose=np.zeros(33*3)# np.zeros it creates an array of zeros we use if for a starting value 3* for axes x,y,z 
    face=np.zeros(478*3)# 478 landmarks on the face
    righthand=np.zeros(21*3)# 21 landmarks on the hand
    lefthand=np.zeros(21*3)# 21 landmarks on the hand

    if results.pose_landmarks:#we will use lstm for pattern recognition so we need to extract the keypoints
        pose=np.array([[res.x,res.y,res.z] for res in results.pose_landmarks.landmark]).flatten()
        # This takes the x, y, z values from all pose landmarks in results
        # It puts them into a 2D array (landmarks × coordinates)
        # Then .flatten() converts it into a 1D vector for the neural network



    if results.face_landmarks:
        face=np.array([[res.x,res.y,res.z] for res in results.face_landmarks.landmark]).flatten()


    if results.left_hand_landmarks:
        lefthand=np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten()


    if results.right_hand_landmarks:
        righthand=np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    npy_path = f'MP_Data/{action}/{sequence}/{frame}.npy'#MP_Data/hello/1/0.npy
    np.save(npy_path, keypoints)
