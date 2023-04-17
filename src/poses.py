import cv2 as cv
import mediapipe as mp
import numpy as np
import pandas as pd

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# setting feature columns for model prediction
columns = []
for val in range(1, 502):
    columns.append(f'x{val}')
    columns.append(f'y{val}')
    columns.append(f'z{val}')
    columns.append(f'v{val}')


def process_pose(frame, results, model, face, height, width):
    # 1. Draw face landmarks
    mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                )

    # 2. Right hand
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                )

    # 3. Left Hand
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                )

    # 4. Pose Detections
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )

    try:
        # Extract Pose landmarks
        pose = results.pose_landmarks.landmark
        pose_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()

        # Extract Face landmarks
        face_row = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten()

        # Concate rows
        row = np.concatenate((pose_row, face_row))

        # Make Detections
        X = pd.DataFrame([row], columns=columns)

        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        # Grab ear coords
        coords = tuple(np.multiply(
            np.array(
                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
            , [width, height]).astype(int))

        cv.rectangle(frame,
                        (coords[0], coords[1] + 5),
                        (coords[0] + len(body_language_class) * 20, coords[1] - 30),
                        (245, 117, 16), -1)
        cv.putText(frame, body_language_class, coords,
                    cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # Get status box
        cv.rectangle(frame, (0, 0), (250, 60), (245, 117, 16), -1)

        # Display Class
        cv.putText(frame, 'CLASS'
                    , (95, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(frame, body_language_class.split(' ')[0]
                    , (90, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

        # # Display Probability
        cv.putText(frame, 'PROB'
                    , (15, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
        cv.putText(frame, str(round(body_language_prob[np.argmax(body_language_prob)], 2))
                    , (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

    except:
        pass

    return frame, body_language_class.split(' ')[0]
