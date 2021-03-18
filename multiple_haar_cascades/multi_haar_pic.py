import cv2

DEBUG = 0

face_cascade = cv2.CascadeClassifier()
hand_cascade = cv2.CascadeClassifier()

if not hand_cascade.load('fist.xml'):
    print('Error loading hand cascade')
    exit(0)

if not face_cascade.load('haarcascade_frontalface_default.xml'):
    print('Error loading face cascade')
    exit(0)

path = "haartest3"
img = cv2.imread(path + ".jpeg")
if img is None:
    print("Error loading image.")
    exit()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#gray = cv2.resize(gray, (480,480))

#cv2.imshow("aaa", gray)

hands = hand_cascade.detectMultiScale(gray, 1.01, 1, None, (300,300))
faces = face_cascade.detectMultiScale(gray, 1.01, 5, cv2.CASCADE_DO_ROUGH_SEARCH,(200,200))

imgres = img.copy()
for (x,y,w,h) in hands:
    center = (x + w//2, y + h//2)
    if x<img.shape[0]/2:
        cv2.ellipse(imgres, center, (w//2, h//2), 0, 0, 360, (255, 0, 0), 4)
    if DEBUG:
        print(hand_xyz[i])

for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    cv2.ellipse(imgres, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    if DEBUG:
        print(face_xyz[i])

cv2.imshow("Source", cv2.resize(img,(480,480)))
cv2.imshow("Result", cv2.resize(imgres,(480,480)))
cv2.imwrite(path + "res.jpeg", imgres)
while True:
    if cv2.waitKey(1) == ord('q'):
        exit()

