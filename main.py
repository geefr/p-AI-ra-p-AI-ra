import cv2
import numpy as np
import mediapipe as mp

import time
from copy import copy, deepcopy
import glm
import math

from pynput.keyboard import Key, Controller
keyboard = Controller()

def calc_forward_vec(left_hip, right_hip):
    l2rhip = right_hip - left_hip
    l2rhip.z = 0.0
    l2rhip = glm.normalize(l2rhip)
    up = glm.vec3(0.0, 0.0, 1.0)

    forward_vec = glm.cross(l2rhip, up)
    forward_vec = glm.normalize(forward_vec)
    # print(f"Forward: {forward_vec}")
    return forward_vec

def calc_arm_angle(forward_vec, arm_vec):
    try:
        forward_vec = copy(forward_vec)
        arm_vec = copy(arm_vec)

        forward_vec.z = 0.0
        arm_vec.z = 0.0

        forward_vec = glm.normalize(forward_vec)
        arm_vec = glm.normalize(arm_vec)

        # Calculate angle between forward and arm
        theta = math.acos(glm.dot(forward_vec, arm_vec))

        plane_normal = glm.vec3(0.0, 0.0, 1.0)
        cross = glm.cross(forward_vec, arm_vec)
        if glm.dot(plane_normal, cross) < 0:
            theta = - theta

        theta = math.degrees(theta)
        return theta
    except Exception:
        return -180

class ButtonState:
    def __init__(self) -> None:
        self.left_arm = False
        self.right_arm = False
        self.pressed = False
    
    def __repr__(self) -> str:
        return f"LEFT: {self.left_arm} RIGHT: {self.right_arm} PRESSED: {self.pressed}"

def init_button_state():
    return {
        'z': ButtonState(),
        'x': ButtonState(),
        'c': ButtonState(),
        'v': ButtonState(),
        'b': ButtonState(),
    }
button_state = init_button_state()

class AngleRange:
    def __init__(self, minA, maxA) -> None:
        self.min_angle = minA
        self.max_angle = maxA

def press_buttons(left_angle, right_angle, button_state):
    button_angles = {
        'z': AngleRange(-90, -60),
        'x': AngleRange(-60, -30),
        'c': AngleRange(-30, 30),
        'v': AngleRange(30, 60),
        'b': AngleRange(60, 90),
    }

    # print(f"press_buttons: {left_angle}, {right_angle}")

    for (button, angles) in button_angles.items():
        state = button_state[button]

        if left_angle >= angles.min_angle and left_angle <= angles.max_angle:
            # print(f"LEFT ACTIVE: {button}")
            state.left_arm = True
        else:
            state.left_arm = False

        if right_angle >= angles.min_angle and right_angle <= angles.max_angle:
            # print(f"RIGHT ACTIVE: {button}")
            state.right_arm = True
        else:
            state.right_arm = False

   #  print(f"0: {button_state['x']}")
    for (button, state) in button_state.items():
        if state.left_arm or state.right_arm:
            if not state.pressed:
                state.pressed = True
                keyboard.press(button)
                print(f"PRESS {button}")
        else:
            if state.pressed:
                state.pressed = False
                keyboard.release(button)
                print(f"RELEASE {button}")
                
    # print(f"1: {button_state['x']}")


# I dunno some magic for game input ;)
#keyboard.press('a')
#keyboard.release('a')


def main():
    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    width = 800
    height = 600
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detection_window = 'detection'
    annotation_window = 'annotation'

    # https://levelup.gitconnected.com/hand-tracking-module-using-python-eb27c8772664
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=False,  # Generate a segmentation mask - don't need it here
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )

    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    first_frame_gray = None

    last_frame = time.time_ns()
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            done = True
            break
        # Output frame to display/annotate
        orig_frame = deepcopy(frame)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

        if first_frame_gray is None:
            first_frame_gray = deepcopy(frame_gray)

        frame_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)

        landmarks = {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
        }
        
        # 3D maths, whee!
        pose_results = pose.process(frame_rgb)
        if pose_results.pose_world_landmarks is not None:
            poses = {}
            for (name, id) in landmarks.items():
                p = pose_results.pose_world_landmarks.landmark[id]
                if p.visibility:
                    poses[name] = glm.vec3(p.x, p.y, p.z)
                else:
                    poses[name] = None
            
            if None in poses.values():
                print("Invalid pose data - Missing a landmark")
                continue

            forward_vec = calc_forward_vec(poses['left_hip'], poses['right_hip'])
            left_arm_vec = poses['left_wrist'] - poses['left_shoulder']
            right_arm_vec = poses['right_wrist'] - poses['right_shoulder']

            left_angle = calc_arm_angle(forward_vec, left_arm_vec)
            right_angle = calc_arm_angle(forward_vec, right_arm_vec)

            press_buttons(left_angle, right_angle, button_state)

        # Preview window
        if pose_results.pose_landmarks is not None:
            for (mark, id) in landmarks.items():
                mark_pos = pose_results.pose_landmarks.landmark[id]

                height, width, channels = frame.shape
                x = int(mark_pos.x * width)
                y = int(mark_pos.y * height)

                cv2.circle(frame, (x,y), 10, (255, 0, 255), cv2.FILLED)

            mp_draw.draw_landmarks(
                frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style()
            )

        cv2.namedWindow(annotation_window)
        cv2.imshow(annotation_window, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True

        next_frame = last_frame + 30e3
        to_sleep = next_frame - time.time_ns()
        if to_sleep > 0:
            time.sleep(to_sleep * 1e6)
        last_frame = next_frame


if __name__ == '__main__':
    main()
