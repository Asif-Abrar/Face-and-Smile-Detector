import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

count = 0
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Can not open camera!")
    exit(0)

while camera.isOpened():
    ret, frame = camera.read()

    if not ret:
        print("Can not receive frame!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    i = 0
    # -- Detect faces
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # -- Sub_frame of face
        faceROI = frame[y:y+h, x:x+w]

        i = i + 1

        if len(faceROI) > 0:
            cv2.putText(frame, 'face' + str(i), (x, y), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0))

        faceROI_gray = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)

        # -- In each face, detect smiles
        smiles = smile_cascade.detectMultiScale(faceROI_gray, scaleFactor=1.7, minNeighbors=20)
        for (x2, y2, w2, h2) in smiles:
            cv2.rectangle(faceROI, (x2, y2), (x2+w2, y2+h2), (0, 0, 255), 4)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0))

    cv2.imshow("Camera", frame)

    if cv2.waitKey(25) == ord("s"):
        cv2.imwrite("test{:>05}.png".format(count), frame)
        print("Image saved test{:>05}.png".format(count))
        count = count + 1

    if cv2.waitKey(50) == ord("q"):
        print("Exiting...")
        break

camera.release()
cv2.destroyAllWindows()
