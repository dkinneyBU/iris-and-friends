import cv2
import pyttsx3

# Initialize face detector and text-to-speech engine
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
engine = pyttsx3.init()

# Optional: configure voice
engine.setProperty('rate', 150)

# Iris-like response
def iris_response(text):
    print(f"Iris: {text}")
    engine.say(text)
    engine.runAndWait()

# Start video capture
cap = cv2.VideoCapture(0)

face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0 and not face_detected:
        iris_response("Ah, there you are. I see you.")
        face_detected = True
    elif len(faces) == 0 and face_detected:
        iris_response("Where did you go?")
        face_detected = False

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('Iris Vision', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
