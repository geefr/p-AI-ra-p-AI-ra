import sys
import cv2
import numpy as np
import mediapipe as mp

import time
from copy import copy, deepcopy
import glm
import math
from PIL import Image, ImageDraw

from pynput.keyboard import Key, Controller
keyboard = Controller()

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
ENABLE_OVERHEAD_VIEW = False
ENABLE_PREVIEW = True

# Enable to log angles
# Stand with chest forward, each arm to front / side in turn
# Adjust offsets to they swepp 0 - 90 degrees, or as close as you can get
CALIBRATION_MODE = False
angle_offset_left = 0
angle_offset_right = 30
# The minimum distance away from center of body to count as active
min_wrist_distance = 0.2
input_mash_time = 0.05
arm_level_thresh = 0.4

up = glm.vec3(0.0, 1.0, 1.0)

def calc_forward_vec(left_hip, right_hip):
    l2rhip = right_hip - left_hip
    l2rhip.z = 0.0
    l2rhip = glm.normalize(l2rhip)

    forward_vec = glm.cross(l2rhip, up)
    forward_vec = glm.normalize(forward_vec)
    # print(f"Forward: {forward_vec}")
    return forward_vec

def is_arm_level(shoulder_pos, arm_pos):
    # Threshold height from shoulder (this is dumb, use angles?)
    d = abs(arm_pos.y - shoulder_pos.y)
    return d < arm_level_thresh
        
def calc_arm_angle(forward_vec, arm_vec):
    try:
        forward_vec = copy(forward_vec)
        arm_vec = copy(arm_vec)

        forward_vec.y = 0.0
        arm_vec.y = 0.0

        forward_vec = glm.normalize(forward_vec)
        arm_vec = glm.normalize(arm_vec)

        # Calculate angle between forward and arm
        theta = math.acos(glm.dot(forward_vec, arm_vec))

        cross = glm.cross(forward_vec, arm_vec)
        if glm.dot(up, cross) < 0:
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
        self.toggle_time = time.time()
    
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
        'z': AngleRange(-150, -50),
        'x': AngleRange(-60, -30),
        'c': AngleRange(-40, 40),
        'v': AngleRange(30, 60),
        'b': AngleRange(50, 150),
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
    now = time.time()
    for (button, state) in button_state.items():
        if state.left_arm or state.right_arm:
            if not state.pressed and (now - state.toggle_time) > input_mash_time:
                state.pressed = True
                state.toggle_time = now
                keyboard.press(button)
                # print(f"PRESS {button}")
            elif (now - state.toggle_time) > input_mash_time:
                state.pressed = False
                state.toggle_time = now
                keyboard.release(button)
        else:
            if state.pressed:
                state.pressed = False
                state.toggle_time = now
                keyboard.release(button)
                # print(f"RELEASE {button}")
                
    # print(f"1: {button_state['x']}")

def print_state(left_angle, right_angle, button_state):
    msg = ""
    msg += f"{int(left_angle):5} "
    for s in button_state.values():
        if s.pressed:
            msg += "x"
        else:
            msg += "_"
    msg += f" {int(right_angle):5}"
    print(msg)

def draw_overhead_view(poses):
    world_min = glm.vec3(-0.5, -0.5, -0.5)
    world_max = glm.vec3(0.5, 0.5, 0.5)
    world_center = glm.vec3(0.0, 0.0, 0.0)

    dim = 800
    pxPerMeter = (world_max - world_min) / dim
    img = Image.new('RGB', (dim, dim), color = 'black')
    draw = ImageDraw.Draw(img)

    # print(poses)

    for (name, pose) in poses.items():
        d = (dim / 2) + ((pose - world_center) * pxPerMeter)
        draw.ellipse([(d.x - 5, d.z - 5),(d.x + 5, d.z + 5)], fill = 'pink', outline='pink')
    
    cvImg = np.array(img) 
    # Convert RGB to BGR 
    cvImg = cvImg[:, :, ::-1].copy() 
    cv2.namedWindow('overhead')
    cv2.imshow('overhead', cvImg)
    if cv2.waitKey(1) == 'q':
        sys.exit()

def main():
    print("Init Camera")

    if sys.platform == 'win32':
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)

    if cap is None or not cap.isOpened():
        print("ERROR: Failed to open camera")
        return 1

    annotation_window = 'annotation'

    # https://levelup.gitconnected.com/hand-tracking-module-using-python-eb27c8772664
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=2,
        enable_segmentation=True,  # Generate a segmentation mask
        min_detection_confidence=0.9,
        min_tracking_confidence=0.9
    )

    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles

    bg_image = None
    first_frame = True

    print("Begin Main Loop")
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            done = True
            break

        if first_frame:
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = (192, 192, 192)
            first_frame = False

        # Output frame to display/annotate
        orig_frame = deepcopy(frame)
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

            # forward_vec = calc_forward_vec(poses['left_hip'], poses['right_hip'])
            forward_vec = calc_forward_vec(poses['left_shoulder'], poses['right_shoulder'])

            if abs(poses['left_wrist'].x) < min_wrist_distance and abs(poses['left_wrist'].z) < min_wrist_distance:
                left_angle = -180
            elif not is_arm_level(poses['left_shoulder'], poses['left_wrist']):
                left_angle = -180
            else:
                left_arm_vec = poses['left_wrist'] - poses['left_shoulder']
                left_angle = calc_arm_angle(forward_vec, left_arm_vec)
                left_angle += angle_offset_left

            if abs(poses['right_wrist'].x) < min_wrist_distance and abs(poses['right_wrist'].z) < min_wrist_distance:
                right_angle = -180
            elif not is_arm_level(poses['right_shoulder'], poses['right_wrist']):
                right_angle = -180
            else:
                right_arm_vec = poses['right_wrist'] - poses['right_shoulder']
                right_angle = calc_arm_angle(forward_vec, right_arm_vec)
                right_angle += angle_offset_right

            # xprint(f"LEFT: {left_angle} RIGHT: {right_angle}")

            press_buttons(left_angle, right_angle, button_state)
            if CALIBRATION_MODE:
                print_state(left_angle, right_angle, button_state)

            if ENABLE_OVERHEAD_VIEW:
                draw_overhead_view(poses)

        if ENABLE_PREVIEW:
            # Preview window
            if pose_results.pose_landmarks is not None:
                # for (mark, id) in landmarks.items():
                #     mark_pos = pose_results.pose_landmarks.landmark[id]
                #     height, width, channels = frame.shape
                #     x = int(mark_pos.x * width)
                #     y = int(mark_pos.y * height)
                #     cv2.circle(frame, (x,y), 10, (255, 0, 255), cv2.FILLED)

                mp_draw.draw_landmarks(
                    frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_draw_styles.get_default_pose_landmarks_style()
                )

            if pose_results.segmentation_mask is not None:
                # Draw segmentation on the image.
                # To improve segmentation around boundaries, consider applying a joint
                # bilateral filter to "results.segmentation_mask" with "image".
                condition = np.stack((pose_results.segmentation_mask,) * 3, axis=-1) > 0.1
                frame = np.where(condition, frame, bg_image)
            else:
                frame = bg_image
            
            cv2.namedWindow(annotation_window)
            cv2.imshow(annotation_window, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                done = True


if __name__ == '__main__':
    main()
