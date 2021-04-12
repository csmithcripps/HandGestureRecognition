import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from collections import deque

import handRepresentation

from csv import DictWriter

def append_dict_as_row(file_name, dict_of_elem, field_names):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        # Add dictionary as wor in the csv
        dict_writer.writerow(dict_of_elem)

    
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (30,30)
fontScale              = 1
fontColor              = (255,0,0)
lineType               = 2

file_name = 'GestureData.csv'
fieldNames = ['Gesture',
                    'Thumb0',
                    'Thumb1',
                    'Thumb2',
                    'Pointer0',
                    'Pointer1',
                    'Pointer2',
                    'Index0',
                    'Index1',
                    'Index2',
                    'Ring0',
                    'Ring1',
                    'Ring2',
                    'Pinky0',
                    'Pinky1',
                    'Pinky2']


gesturesToCapture = ['0','1','2','3','4','5','6','7','8','9']
gestureIDX = 0

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(    
    min_detection_confidence=0.7,
    min_tracking_confidence=0.1) as hands:
  while cap.isOpened():
    gestureName = gesturesToCapture[gestureIDX]

    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    newRow = {}
    if results.multi_hand_landmarks:
        # print(HandAveCoordinates)
        handedness = [
          handedness.classification[0].label
          for handedness in results.multi_handedness
        ]
        hand0 = handRepresentation.hand(results.multi_hand_landmarks[0],
                                      handedness[0])
        newRow = {fieldNames[0]:gestureName}
        newRow.update(hand0.JointAngles)
        # print(results, '\n')

        cv2.putText(image,gestureName, 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('MediaPipe Hands', image)
    # break
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
      break
    if key == ord('n'):
      if gestureIDX < len(gesturesToCapture):
        gestureIDX +=1
      else:
        gestureIDX = 0
    if key == 32:
        print("Captured Angles")
        if newRow:
            print(newRow, '\n')
            append_dict_as_row(file_name,newRow,fieldNames)

            
cap.release()

