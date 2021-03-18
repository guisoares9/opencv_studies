import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

DEBUG = 0

def getXYZ( xProj, yProj, focus, hProj, hReal):
    xReal = xProj*hReal/hProj
    yReal = yProj*hReal/hProj
    zReal = focus*hReal/hProj
    return (xReal, yReal, zReal)

head_height = 20
focal_depth = 1000

face_cascade = cv2.CascadeClassifier()
hand_cascade = cv2.CascadeClassifier()

if not hand_cascade.load('fist.xml'):
    print('Error loading hand cascade')
    exit(0)

if not face_cascade.load('haarcascade_frontalface_default.xml'):
    print('Error loading face cascade')
    exit(0)

#vid = cv2.VideoCapture("https://192.168.1.5:8080/video")
vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print('Video error')
    exit(0)

if DEBUG:  
    # Creating figure 
    fig = plt.figure(figsize = (10, 7)) 
    ax = plt.axes(projection ="3d") 
    plt.title("3D Visualization")
    ax.set_xlabel('X-axis', fontweight ='bold')  
    ax.set_ylabel('Y-axis', fontweight ='bold')  
    ax.set_zlabel('Z-axis', fontweight ='bold') 

while True:

    ret, frame = vid.read()
    frame = cv2.resize(frame,(360,240))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    hands = []
    faces = []
    hands = hand_cascade.detectMultiScale(gray, 1.03, 3, cv2.CASCADE_DO_ROUGH_SEARCH, (5,5), None)
    faces = face_cascade.detectMultiScale(gray, 1.03, 3, cv2.CASCADE_DO_ROUGH_SEARCH, (10,10), None)
  

    i = 0
    hand_xyz = np.zeros((50,3))
    for (x,y,w,h) in hands:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
        hand_xyz[i] = getXYZ(x + w/2 - frame.shape[1]/2, frame.shape[0]/2 - y - h/2, focal_depth, h, head_height)
        if DEBUG:
            print(hand_xyz[i])

    i = 0
    face_xyz = np.zeros((50,3))
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        face_xyz[i] = getXYZ(x + w/2 - frame.shape[1]/2, frame.shape[0]/2 - y - h/2, focal_depth, h, head_height)
        if DEBUG:
            print(face_xyz[i])
            ax.scatter3D(face_xyz[i][0],face_xyz[i][1],face_xyz[i][2], color = "blue")
        
        
        
    cv2.imshow('Capture - Face detection', frame)
    if DEBUG:
        plt.show(block=False) 
        plt.pause(0.001)
    if cv2.waitKey(1) == 27:
        break

vid.release()
cv2.destroyAllWindows()
