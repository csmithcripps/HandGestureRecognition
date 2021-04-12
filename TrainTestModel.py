# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

import handRepresentation


   
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,0,255)
lineType               = 2


gestureThreshold = 0.8
# Split-out validation dataset
names = ['Gesture',
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
                    
dataset = read_csv('./GestureData.csv', names=names)

array = dataset.values
X = array[:,1:17]
y = array[:,0]
print(y)


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


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
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.rectangle(image,(0,0),(800,40),(0,0,0),-1)
    if results.multi_hand_landmarks:
        # print(HandAveCoordinates)
        hand0 = handRepresentation.hand(results.multi_hand_landmarks[0],
                                      results.multi_handedness[0])
        newRow = {names[1]:hand0.FrameAngle,}
        newRow.update(hand0.JointAngles)
        # print(newRow, '\n')
        X_test = np.array(list(newRow.values()))

        predictedGesture = model.predict(X_test.reshape(1,-1))
        proba = model.predict_proba(X_test.reshape(1,-1))
        probability = (max(proba[0]))
        if max(proba[0])>gestureThreshold :
            print(predictedGesture[0])
        else:
            print('\nNoGesture')


        bottomLeftCornerOfText = (10,30)
        cv2.putText(image,str(predictedGesture[0]), 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        bottomLeftCornerOfText = (100,30)
        cv2.putText(image,str(100*probability) + '%', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        bottomLeftCornerOfText = (10,30)
        cv2.putText(image, 'No Hand', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
    cv2.imshow('MediaPipe Hands', image)
    # break
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
      break

    if key == 32:
        print("Captured Angles")
        if newRow:
            print(newRow, '\n')
            append_dict_as_row(file_name,newRow,fieldNames)

            
cap.release()


