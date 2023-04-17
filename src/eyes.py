import cv2 as cv
import numpy as np
import utils
from facemesh_indices import *
from configs import *

# Euclaidean distance
def euclaideanDistance(point, point1):
    return np.linalg.norm(np.array(point) - np.array(point1))


# landmark detection function
def landmarksDetection(results, img_height, img_width):
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.face_landmarks.landmark]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Blinking Ratio
def blinkRatio(landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


def get_eye_direction(eye_points, eye_center, frame):
    calculated_eye_center = np.mean(eye_points, axis=0)
    eye_direction =  calculated_eye_center - eye_center

    pos_eye = None

    if -15 < eye_direction[0] < 15:
        pos_eye = "CENTER"
        color = [utils.YELLOW, utils.PINK]
    elif eye_direction[0] <= 15:
        pos_eye = "RIGHT"
        color = [utils.GRAY, utils.YELLOW]
    elif eye_direction[0] >= 15:
        pos_eye = "LEFT"
        color = [utils.GRAY, utils.YELLOW]
    
    # draw calculated_eye_center
    cv.circle(frame, (int(calculated_eye_center[0]), int(calculated_eye_center[1])), 4, utils.BLUE, 2)

    # draw line from eye center to 
    cv.line(frame, (int(eye_center[0]), int(eye_center[1])), (int(calculated_eye_center[0]), int(calculated_eye_center[1])), utils.WHITE, 1, cv.LINE_AA)

    return pos_eye, color


def process_eyes(frame, results, CEF_COUNTER, TOTAL_BLINKS, closed_eyes_frames, frame_height=0, frame_width=0, ):
    mesh_coords = landmarksDetection(results, frame_height, frame_width)
    ratio = blinkRatio(mesh_coords, RIGHT_EYE, LEFT_EYE)
    utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                    utils.YELLOW)
    
    if ratio > BLINK_THRESHOLD:
        CEF_COUNTER += 1
        closed_eyes_frames += 1
        utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                                        pad_x=6, pad_y=6, )
    else:
        if CEF_COUNTER > CLOSED_EYES_FRAME:
            TOTAL_BLINKS += 1
            CEF_COUNTER = 0

    utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

    # drawing eye centers
    for idx, point in enumerate(mesh_coords):
        if idx in [*LEFT_EYE_CENTER, *RIGHT_EYE_CENTER]:
            cv.circle(frame, point, 2, utils.RED, -1)

    right_coords = [mesh_coords[p] for p in RIGHT_EYE]
    left_coords = [mesh_coords[p] for p in LEFT_EYE]
    
    eye_position, color = get_eye_direction(right_coords, mesh_coords[RIGHT_EYE_CENTER[0]], frame)
    utils.colorBackgroundText(frame, f'R: {eye_position}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)

    eye_position_left, color = get_eye_direction(left_coords, mesh_coords[LEFT_EYE_CENTER[0]], frame)
    utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)

    return frame, CEF_COUNTER, TOTAL_BLINKS, eye_position, closed_eyes_frames