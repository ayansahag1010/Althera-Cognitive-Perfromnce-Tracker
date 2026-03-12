import cv2
import csv
from datetime import datetime

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

with open("emotion_results.csv","a",newline="") as f:

    writer = csv.writer(f)
    writer.writerow(["timestamp","emotion"])

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        emotion = "neutral"

        for (x,y,w,h) in faces:

            # simple rule-based emotion detection
            if h > 250:
                emotion = "happy"
            elif h < 120:
                emotion = "sad"
            else:
                emotion = "neutral"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        timestamp = datetime.now()
        writer.writerow([timestamp,emotion])

        cv2.putText(
            frame,
            emotion,
            (50,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

        cv2.imshow("Emotion Detection",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()