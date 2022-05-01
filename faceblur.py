import cv2 as cv

faceCascade = cv.CascadeClassifier('E:\PythonCodes\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
video_capture = cv.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    faces = faceCascade.detectMultiScale(frame, 1.2, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        face_roi = frame[y:y+h, x:x+w]
        blur = cv.GaussianBlur(face_roi, (91, 91), 0)
        frame[y:y+h, x:x+w] = blur
    if faces == ():
        cv.putText(frame, 'No Face Found!!', (20, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
    cv.imshow('Blur Face', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
video_capture.release()
cv.destroyAllWindows()