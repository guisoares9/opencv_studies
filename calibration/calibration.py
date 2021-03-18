import numpy as np
import cv2 as cv
import time

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

square_size = 28
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
#print(objp)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = square_size * objp
#print(objp)
#exit(0)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
    
for i in range(9,50):

    img = cv.imread("patternphotos/calib" + str(i) + ".png")

    #if img == None:
    #    print("Error while geting image \n")
    #    exit(0)

    cv.imshow("a", img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

        # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners

        cv.drawChessboardCorners(img, (9,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

    time.sleep(0.001)

cv.destroyAllWindows()

print("Inicializing calibration... \n")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print("Calibration finished... \n")

cam_file_name = str(input("Insira o nome do arquivo destinado a matriz da camera: "))

print("Saving camera matrix... \n")

camfile = open(cam_file_name,"w")

for i in range(3):
    for j in range(3):
        camfile.write(str(mtx[i][j]) + " ")
    camfile.write("\n")
camfile.write("\n")
for i in range(5):
    camfile.write(str(dist[0][i]) + " ")

camfile.close()
camfile = open(cam_file_name, "r")
print(camfile.read())


print(mtx)
print(dist)
#print(rvecs)
#print(tvecs)
