import cv2 as cv
import pandas as pd

import mediapipe as mp
import time
import pickle

from facemesh_indices import *

from eyes import process_eyes
from poses import process_pose

import utils
from configs import *

from plotter import plot


def main():
    # loading model
    with open('body_language.pkl', 'rb') as f:
        model = pickle.load(f)

    # variables
    frame_counter =0
    closed_eyes_frames = 0
    CEF_COUNTER = 0
    TOTAL_BLINKS = 0

    map_face_mesh = mp.solutions.holistic

    # uncomment to use video file
    # FILE = "video1.mp4"
    # FILE = "video2.mp4"
    # FILE = "video3.mp4"
    FILE = "blink.mp4"

    cap = cv.VideoCapture("..\\videos\\"+FILE)

    # uncomment to use webcam
    # cap = cv.VideoCapture(0)

    data = []
    fps_history = []

    start_time = time.time()

    with map_face_mesh.Holistic(min_detection_confidence=0.5, refine_face_landmarks=True, model_complexity=1) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            frame_counter += 1 

            if not ret:
                break

            frame_height, frame_width, _ = frame.shape

            results = holistic.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            if results.face_landmarks:
                # processing eyes position
                frame, CEF_COUNTER, TOTAL_BLINKS, eye_position, closed_eyes_frames = process_eyes(frame, results, CEF_COUNTER, TOTAL_BLINKS, closed_eyes_frames, frame_height, frame_width)
                
                # filtering out iris landmarks
                results_without_iris = []

                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    if idx not in [*LEFT_IRIS, *RIGHT_IRIS]:
                        results_without_iris.append(landmark)

                # processing poses
                frame, pose = process_pose(frame, results, model, results_without_iris, frame_height, frame_width)

                # saving data
                data.append([pose, CEF_COUNTER, TOTAL_BLINKS, eye_position])

            # calculating fps and appending to fps_history
            end_time = time.time() - start_time
            fps = frame_counter / end_time
            fps_history.append(fps)

            frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (frame_width - 140, 50), bgOpacity=0.9,
                                             textThickness=2)

            # making video easier to see (0.75%) usefull when capture resoloution is high                        
            frame = cv.resize(frame, None, fx=0.75, fy=0.75)

            cv.imshow('frame', frame)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break


    cap.release()
    cv.destroyAllWindows()

    # plotting stats
    plot(data, fps_history, closed_eyes_frames)

    # saving data to csv
    df = pd.DataFrame(data, columns=['pose', 'cef_counter', 'total_blinks', 'eye_position'])
    df.to_csv('..\\outputs\\data.csv', index=False)



if __name__ == "__main__":
    main()
