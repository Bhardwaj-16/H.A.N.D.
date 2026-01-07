import cv2
import mediapipe as mp

from packages.hand_landmarks import get_frame_and_landmarks
from packages.gestures import detect_gesture

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while cap.isOpened():
    frame, result = get_frame_and_landmarks(cap)
    if frame is None:
        break

    gesture = "NONE"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp.solutions.hands.HAND_CONNECTIONS
            )
            gesture = detect_gesture(hand_landmarks)

    cv2.putText(
        frame,
        gesture,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("H.A.N.D.", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()