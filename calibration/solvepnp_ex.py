import cv2 as cv
import cv2.aruco
import numpy as np

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

mtx =   [[725.7657025001567, 0.0, 302.64698191622074],
        [0.0, 733.3294790963405, 277.5564048217287],
        [0.0, 0.0, 1.0] 
        ]

dist =  [[0.35136313880105813, -3.2814702001594096, -0.002180299088962941, 0.00383256440875228, 12.160646308561365 ]]

mtx = np.array(mtx)
dist = np.array(dist)

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

square_size = 28

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = square_size * objp

print(objp)

axis = np.float32([[300,0,0], [0,300,0], [0,0,-300]]).reshape(-1,3)

axisCube = np.float32([[0,0,0], [0,150,0], [150,150,0], [150,0,0],
                   [0,0,-150],[0,150,-150],[150,150,-150],[150,0,-150] ])

ARUCO_PARAMETERS = cv.aruco.DetectorParameters_create()
ARUCO_DICT = cv.aruco.Dictionary_get(cv.aruco.DICT_4X4_50)


vid = cv.VideoCapture(0)

if not vid.isOpened():
    print("Grabing video error.")
    exit(0)

i = 0
while True:

    ret_vid, img = vid.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret_chess, corners = cv.findChessboardCorners(gray, (9,6),None)

    aruco_corners, aruco_ids, rejectedArucoPoints = cv.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMETERS)
    print(aruco_corners)
    print(aruco_ids)
    if aruco_ids is not None:
        img = cv.aruco.drawDetectedMarkers(img, aruco_corners, aruco_ids, (0,0,255))
        for i in range(len(aruco_ids)):
            print(i)
            print(int(i))
            print(len(aruco_ids))
            aruco_rvec, aruco_tvec, _ = cv.aruco.estimatePoseSingleMarkers(aruco_corners[int(i)], 75, mtx, dist)
            img = cv.aruco.drawAxis(img, mtx, dist, aruco_rvec, aruco_tvec, 100)

    if ret_chess == True:
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        imgptsCube, jac = cv.projectPoints(axisCube, rvecs, tvecs, mtx, dist)
        img = draw(img, corners2, imgpts)
        img = drawCube(img, corners2, imgptsCube)
        cv.imshow('img',img)

        
    cv.imshow('img',img)

    k = cv.waitKey(1) & 0xFF

    if k == ord('s'):
            cv.imwrite('AR'+str(1)+'.png', img)
            i += 1
    if k == ord('q'):
            exit(0)

cv.destroyAllWindows()