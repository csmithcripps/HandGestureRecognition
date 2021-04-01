import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set up a queue for the location of the hand Base
HandAveCoordinates = deque([], maxlen = 50)

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(    
    min_detection_confidence=0.7,
    min_tracking_confidence=0.1) as hands:
  while cap.isOpened():
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
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            x = np.average([landmark.x for landmark in landmarks.landmark])
            y = np.average([landmark.y for landmark in landmarks.landmark])
        HandAveCoordinates.append([x,y])
        print(HandAveCoordinates)
        for coord in HandAveCoordinates:
            xInt = int(coord[0]*image.shape[1])
            yInt = int(coord[1]*image.shape[0])
            cv2.circle(image, (xInt,yInt), 2, (255,0,0), 2)
    #   for hand_landmarks in results.multi_hand_landmarks:
    #     mp_drawing.draw_landmarks(
    #         image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        HandAveCoordinates = deque([], maxlen = 50)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(1) & 0xFF == 27:
      break
cap.release()
