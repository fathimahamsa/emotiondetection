import cv2
from deepface import DeepFace
from gtts import gTTS
import os
from playsound import playsound
# intead gtts use pytts
# Load video from webcam
cap = cv2.VideoCapture(0)

last_emotion = ""   # to avoid repeated TTS for same emotion

while True:
    key, img = cap.read()

    # Analyze emotion
    results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

    # Get dominant emotion
    emotion = results[0]['dominant_emotion']

    # Display emotion on frame
    cv2.putText(img, f'Emotion: {emotion}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Emotion Recognition", img)

    # Play voice **only when emotion changes**
    if emotion != last_emotion:
        last_emotion = emotion
        
        # Convert text to speech
        text = f"You look {emotion}"
        tts = gTTS(text=text, lang='en')
        tts.save("emotion.mp3")

        # Play the audio
        playsound("emotion.mp3")

    # Quit when pressing "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
