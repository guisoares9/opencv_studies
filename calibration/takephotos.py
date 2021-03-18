import cv2 as cv

vid = cv.VideoCapture(0)
if not vid.isOpened:
    exit()

i = 0
while True:

    ret, frame = vid.read()
    cv.imshow("a",frame)
    cv.waitKey(1)
    #com = cv.waitKey(1) 
    if cv.waitKey(1) == 27:
        cv.imwrite("calib" + str(i) + ".png", frame)
        i+=1

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()