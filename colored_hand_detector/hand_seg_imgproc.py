import imutils
import cv2
import time

def nothing(val):
    pass

print("OpenCV version: " + cv2.__version__)

#vid = cv2.VideoCapture("http://192.168.1.11:8080/video")
vid = cv2.VideoCapture(0)

print(vid.isOpened())

if not vid.isOpened():
    print("exit by video initializing error")
    exit(0)

trackbar_window_name = 'Thresh Trackbar'

trackbar_flagGRAY_name = 'Thresh GRAY Flag'
trackbar_flagHSV_name = 'Thresh HSV Flag'
trackbar_thresh_name = 'Thresh GRAY'
trackbar_thresh_name_Hmin = 'HSV: min H'
trackbar_thresh_name_Smin = 'HSV: min S'
trackbar_thresh_name_Vmin = 'HSV: min V'
trackbar_thresh_name_Hmax = 'HSV: max H'
trackbar_thresh_name_Smax = 'HSV: max S'
trackbar_thresh_name_Vmax = 'HSV: max V'

thresh_methodHSV_flag = False
thresh_methodGRAY_flag = False
thresh_gray_value = 45
thresh_Hmin_value = 0
thresh_Hmax_value = 181
thresh_Smin_value = 0
thresh_Smax_value = 13
thresh_Vmin_value = 227
thresh_Vmax_value = 255


cv2.namedWindow(trackbar_window_name,cv2.WINDOW_FREERATIO)

cv2.createTrackbar(trackbar_flagGRAY_name, trackbar_window_name, thresh_methodGRAY_flag, 1, nothing)
cv2.createTrackbar(trackbar_flagHSV_name, trackbar_window_name , thresh_methodHSV_flag, 1, nothing)
cv2.createTrackbar(trackbar_thresh_name, trackbar_window_name , thresh_gray_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Hmin, trackbar_window_name , thresh_Hmin_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Hmax, trackbar_window_name , thresh_Hmax_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Smin, trackbar_window_name , thresh_Smin_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Smax, trackbar_window_name , thresh_Smax_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Vmin, trackbar_window_name , thresh_Vmin_value, 255, nothing)
cv2.createTrackbar(trackbar_thresh_name_Vmax, trackbar_window_name , thresh_Vmax_value, 255, nothing)

while(True):
    # Capture frame-by-frame
    t = time.time()
    ret, frame = vid.read()
    
    thresh_methodHSV_flag = cv2.getTrackbarPos(trackbar_flagHSV_name,trackbar_window_name)

    thresh_methodGRAY_flag = cv2.getTrackbarPos(trackbar_flagGRAY_name,trackbar_window_name)

    #HSV THRESH
    if thresh_methodHSV_flag:

        
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        #hsv = cv2.fastNlMeansDenoisingColored(hsv,hsv)

        thresh_Hmin_value = cv2.getTrackbarPos(trackbar_thresh_name_Hmin,trackbar_window_name)
        thresh_Smin_value = cv2.getTrackbarPos(trackbar_thresh_name_Smin,trackbar_window_name)
        thresh_Vmin_value = cv2.getTrackbarPos(trackbar_thresh_name_Vmin,trackbar_window_name)
        thresh_Hmax_value = cv2.getTrackbarPos(trackbar_thresh_name_Hmax,trackbar_window_name)
        thresh_Smax_value = cv2.getTrackbarPos(trackbar_thresh_name_Smax,trackbar_window_name)
        thresh_Vmax_value = cv2.getTrackbarPos(trackbar_thresh_name_Vmax,trackbar_window_name)

        thresh_hsv = cv2.inRange(hsv, (thresh_Hmin_value, thresh_Smin_value, thresh_Vmin_value), (thresh_Hmax_value, thresh_Smax_value, thresh_Vmax_value))
       
        #thresh_hsv = cv2.erode(thresh_hsv, None)
        #thresh_hsv = cv2.dilate(thresh_hsv, None)
        #thresh_hsv = cv2.morphologyEx(thresh_hsv, cv2.MORPH_OPEN, None)
        thresh_hsv = cv2.morphologyEx(thresh_hsv, cv2.MORPH_DILATE, None)
        thresh_hsv = cv2.morphologyEx(thresh_hsv, cv2.MORPH_CLOSE, None)
        
        

        cnts_hsv, hierarchy = cv2.findContours(thresh_hsv,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        

        filtered = 0
        for i in range(len(cnts_hsv)):
            cnt = cnts_hsv[int(i)]
            cnt = cv2.approxPolyDP(cnt, 0.05*cv2.arcLength(cnt, False), True)
            moment_hsv = cv2.moments(cnt)
            moment_iter = moment_hsv['m00']
            if moment_iter >= 1500: 
                cnts_hsv[i] = cv2.convexHull(cnts_hsv[i])
                pts = cv2.convexHull(cnts_hsv[i],returnPoints= True)
                cv2.drawContours( frame, cnts_hsv, i, ( 0, 0, 255), 5)
                cv2.drawContours( frame, pts, -1, ( 0, 255, 0), 5)
                filtered += 1
       
        print(str(time.time()-t) + ': Higher than limit: ' + str(filtered) + ' , All contours: ' + str(len(cnts_hsv)) + '\n')

        
    #GRAY THRESH
    if thresh_methodGRAY_flag:   
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_gray_value = cv2.getTrackbarPos(trackbar_thresh_name,trackbar_window_name)
        thresh_gray = cv2.threshold(gray, thresh_gray_value, 255, cv2.THRESH_BINARY)[1]

        thresh_gray = cv2.erode(thresh_gray, None, iterations=2)
        thresh_gray = cv2.dilate(thresh_gray, None, iterations=2)

        cnts_gray = cv2.findContours(thresh_gray.copy(), cv2.RETR_EXTERNAL,
	    cv2.CHAIN_APPROX_SIMPLE)

        cnts_gray = imutils.grab_contours(cnts_gray)

        if len(cnts_gray):
            c_gray = max(cnts_gray, key=cv2.contourArea)
            
            extLeft = tuple(c_gray[c_gray[:, :, 0].argmin()][0])
            extRight = tuple(c_gray[c_gray[:, :, 0].argmax()][0])
            extTop = tuple(c_gray[c_gray[:, :, 1].argmin()][0])
            extBot = tuple(c_gray[c_gray[:, :, 1].argmax()][0])

            cv2.drawContours(frame, [c_gray], -1, (0, 255, 255), 2)
            cv2.circle(frame, extLeft, 6, (0, 0, 255), -1)
            cv2.circle(frame, extRight, 6, (0, 255, 0), -1)
            cv2.circle(frame, extTop, 6, (255, 0, 0), -1)
            cv2.circle(frame, extBot, 6, (255, 255, 0), -1)

        

        
        # Our operations on the frame come here
    
    t = time.time() - t
    fps = 1/t
    fps = round(fps,2)
    #print(fps)

    cv2.putText(frame, str(fps) , (50,50), 1, cv2.FONT_HERSHEY_COMPLEX,(0,0,255),1,cv2.LINE_AA)
    # Display the resulting frame
    
    frame = cv2.resize(frame, (480,360))
    cv2.imshow('Frame',frame)

    if thresh_methodHSV_flag: 
        hsv = cv2.resize(hsv, (480,360))
        thresh_hsv = cv2.resize(thresh_hsv, (480,360))
        cv2.imshow('HSV',hsv)
        cv2.imshow('Thresh HSV', thresh_hsv)

    if thresh_methodGRAY_flag:
        thresh_gray = cv2.resize(thresh_gray, (480,360))
        gray = cv2.resize(thresh_gray, (480,360))
        cv2.imshow('thresh',thresh_gray)
        cv2.imshow('gray',gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
vid.release()
cv2.destroyAllWindows()