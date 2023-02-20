import cv2
from Recognizer import Recognizer
from datetime import datetime

recognizer = Recognizer()
recognizer.load_encoding_images()

cap = cv2.VideoCapture(0)
attendance=[]
time=[]

while True:
    ret, frame = cap.read()

    face_locations, face_names = recognizer.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
        if name not in attendance:
            attendance.append(name)
            time.append(datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
print(attendance)
print(time)
recognizer.export_xlsx(attendance,time)
cv2.destroyAllWindows()
