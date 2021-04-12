import mediapipe, math
class hand:
    JointNames = [  'Thumb0',
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

    JointLandmarkIdx = {'Thumb0':   (0,1,2),
                        'Thumb1':   (1,2,3),
                        'Thumb2':   (2,3,4),
                        'Pointer0': (0,5,6),
                        'Pointer1': (5,6,7),
                        'Pointer2': (6,7,8),
                        'Index0':   (0,9,10),
                        'Index1':   (9,10,11),
                        'Index2':   (10,11,12),
                        'Ring0':    (0,13,14),
                        'Ring1':    (13,14,15),
                        'Ring2':    (14,15,16),
                        'Pinky0':   (0,17,18),
                        'Pinky1':   (17,18,19),
                        'Pinky2':   (18,19,20)}


    def __init__(self, landmarks, handedness = 'right'):
        self.handedness = handedness
        self.JointAngles = {}
        self.FrameAngle = 0
        self.PalmFacing = 0
        for joint in self.JointNames:
            self.JointAngles[joint] = 0
        if landmarks:
            self.calculatePose(landmarks)

    def calcSingleAngle(self, prevLandmark,currentLandmark, nextLandmark):
        x1 = currentLandmark.x
        y1 = currentLandmark.y
        z1 = currentLandmark.z
        x2 = prevLandmark.x
        y2 = prevLandmark.y
        z2 = prevLandmark.z
        x3 = nextLandmark.x
        y3 = nextLandmark.y
        z3 = nextLandmark.z
  
        num = (x2-x1)*(x3-x1)+(y2-y1)*(y3-y1)+(z2-z1)*(z3-z1) 
    
        den = math.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)*\
                    math.sqrt((x3-x1)**2+(y3-y1)**2+(z3-z1)**2) 
    
        angle = math.degrees(math.acos(num / den)) 
    
        return round(angle, 3) 


    def calculatePose(self, landmarks):
        landmarkList = list(landmarks.landmark)
        baseX = landmarkList[0].x
        baseY = landmarkList[0].y
        IndexFingerX = landmarkList[12].x
        IndexFingerY = landmarkList[12].y
        self.FrameAngle = math.degrees(math.atan2(IndexFingerY-baseY, IndexFingerX-baseX))
        for joint in self.JointNames:
            previousLndMrk = landmarkList[self.JointLandmarkIdx[joint][0]]
            currentLndMrk = landmarkList[self.JointLandmarkIdx[joint][1]]
            nextLndMrk = landmarkList[self.JointLandmarkIdx[joint][2]]
            self.JointAngles[joint] = self.calcSingleAngle(previousLndMrk, currentLndMrk, nextLndMrk)
        


