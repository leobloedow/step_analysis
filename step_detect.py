import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# --- MediaPipe Initialization ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Input Source Selection ---
print("Choose input source:")
print("1. Use webcam")
print("2. Pick a video file")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    cap = cv2.VideoCapture(0)
elif choice == '2':
    file_path = input("Enter the full path to the video file: ").strip()
    cap = cv2.VideoCapture(file_path) if file_path else None
else:
    print("Invalid choice. Exiting.")
    exit()

if not cap or not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# --- Step Detection Parameters & Variables ---
step_phase = "Searching"
step_count = 0
frames_above_threshold = 0
step_start_time = 0
last_steps = deque(maxlen=15)
distance_buffer = deque(maxlen=5)
prev_distance = 0
cooldown_start_time = 0

# --- TUNABLE PARAMETERS ---
MIN_STEP_THRESHOLD_PIXELS = 60
PEAK_DROP_PERCENTAGE = 0.85
CONFIRMATION_FRAMES = 2
STEP_TIMEOUT_SECONDS = 5.0

video_ended = False

# --- Main Loop ---
while cap.isOpened():
    if video_ended:
        print("Video ended. Press 'q' to quit.")
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        break

    success, frame = cap.read()
    if not success:
        video_ended = True
        continue

    if 'panel_width' not in locals() or 'panel_height' not in locals():
        panel_height, panel_width = frame.shape[:2]

    if choice == '1':
        frame = cv2.flip(frame, 1)

    frame.flags.writeable = False
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    frame.flags.writeable = True

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        landmarks = results.pose_landmarks.landmark
        lm = mp_pose.PoseLandmark

        # Get positions for ankles and foot tips
        left_ankle = landmarks[lm.LEFT_ANKLE.value]
        right_ankle = landmarks[lm.RIGHT_ANKLE.value]
        left_foot_tip = landmarks[lm.LEFT_FOOT_INDEX.value]
        right_foot_tip = landmarks[lm.RIGHT_FOOT_INDEX.value]

        # Check visibility thresholds for reliability
        if (left_ankle.visibility > 0.6 and right_ankle.visibility > 0.6 and
            left_foot_tip.visibility > 0.6 and right_foot_tip.visibility > 0.6):

            # Compute average foot positions (ankle and foot tip) for each foot
            la_x = int(((left_ankle.x + left_foot_tip.x) / 2) * panel_width)
            la_y = int(((left_ankle.y + left_foot_tip.y) / 2) * panel_height)
            ra_x = int(((right_ankle.x + right_foot_tip.x) / 2) * panel_width)
            ra_y = int(((right_ankle.y + right_foot_tip.y) / 2) * panel_height)

            cv2.line(frame, (la_x, la_y), (ra_x, ra_y), (0, 255, 255), 3)

            # Calculate lateral distance between feet
            current_distance = math.hypot(ra_x - la_x, ra_y - la_y)

            # Append to buffer and smooth distance
            distance_buffer.append(current_distance)
            smoothed_distance = sum(distance_buffer) / len(distance_buffer)

            # Calculate velocity (difference in smoothed distance)
            velocity = smoothed_distance - prev_distance
            prev_distance = smoothed_distance

            # Calculate vertical distance difference between feet for vertical movement check
            vertical_distance = abs(la_y - ra_y)

            # Adjusted thresholds
            MIN_STEP_THRESHOLD_PIXELS_ADJ = 60  # base threshold for lateral foot distance
            MIN_VERTICAL_MOVEMENT = 20          # minimal vertical difference to consider foot movement significant
            VELOCITY_THRESHOLD = 0.5            # minimal positive velocity to consider feet moving apart

            # Implement step state machine with smoothing, velocity and vertical checks

            if step_phase == "Searching":
                if (smoothed_distance > MIN_STEP_THRESHOLD_PIXELS_ADJ and
                    velocity > VELOCITY_THRESHOLD and
                    vertical_distance > MIN_VERTICAL_MOVEMENT):
                    frames_above_threshold += 1
                    if frames_above_threshold >= CONFIRMATION_FRAMES:
                        step_phase = "Increasing"
                        step_start_time = time.time()
                        max_dist_in_step = smoothed_distance
                        frames_above_threshold = 0
                else:
                    frames_above_threshold = 0

            elif step_phase == "Increasing":
                frames_above_threshold = 0
                if time.time() - step_start_time > STEP_TIMEOUT_SECONDS:
                    print("USER STOPPED (Timeout)")
                    step_phase = "Searching"
                    continue

                if smoothed_distance > max_dist_in_step:
                    max_dist_in_step = smoothed_distance
                elif smoothed_distance < max_dist_in_step * PEAK_DROP_PERCENTAGE:
                    step_count += 1
                    duration = time.time() - step_start_time
                    step_data = {
                        "id": step_count,
                        "length": max_dist_in_step,
                        "duration": duration
                    }
                    last_steps.append(step_data)
                    print(f"Step #{step_count} | Length: {max_dist_in_step:.0f}px | Duration: {duration:.2f}s")
                    step_phase = "Cooldown"
                    cooldown_start_time = time.time()

            elif step_phase == "Cooldown":
                # Prevent double counting steps too quickly
                COOLDOWN_TIME = 0.05  # seconds
                if time.time() - cooldown_start_time > COOLDOWN_TIME:
                    step_phase = "Searching"

            elif step_phase == "Decreasing":
                if smoothed_distance < MIN_STEP_THRESHOLD_PIXELS_ADJ * 0.9:
                    step_phase = "Searching"
                    max_dist_in_step = 0

    # ================================================================= #
    # --- "LAST 15 STEPS" DISPLAY (remains the same) ---
    # ================================================================= #
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_y = 10
    panel_height = frame.shape[0]
    panel_width = frame.shape[1]

    overlay = frame.copy()

    cv2.putText(frame, f"Passos: {step_count}", (30, panel_y + 60), font, 1.8, (255, 255, 255), 4)
    cv2.line(frame, (20, panel_y + 85), (panel_width - 20, panel_y + 85), (255, 255, 255), 2)

    start_y = panel_y + 180
    line_height = 55
    for i, step in enumerate(reversed(last_steps)):
        y_pos = start_y + i * line_height
        if y_pos > panel_height - 20:
            break
        step_text = f"#{step['id']}:  {step['length']:.0f} px   -   {step['duration']:.2f} s"
        cv2.putText(frame, step_text, (50, y_pos), font, 1.2, (255, 0, 0), 3)

    cv2.imshow('Full Body Pose & Step Analysis', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
print("Exiting...")
cap.release()
cv2.destroyAllWindows()
pose.close()