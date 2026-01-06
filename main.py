import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
import csv
from collections import deque
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime

# ==========================================
# 📂 User Input UI
# ==========================================
def get_video_source():
    root = tk.Tk()
    root.withdraw()
    use_webcam = messagebox.askyesno("Select Input Source", "Do you want to use the WEBCAM?\n\n(Yes = Webcam, No = Video File)")
    if use_webcam:
        root.destroy(); return None 
    else:
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
        root.destroy()
        if not file_path: exit()
        return file_path

VIDEO_SOURCE = get_video_source()

# ==========================================
# 📁 Universal Name
# ==========================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if VIDEO_SOURCE is None:
    folder_name = f"SitToStand_Webcam_{timestamp}"
else:
    base_name = os.path.basename(VIDEO_SOURCE).split('.')[0]
    folder_name = f"SitToStand_{base_name}_{timestamp}"

os.makedirs(folder_name, exist_ok=True)
print(f"📂 Output Folder: {folder_name}")

video_output_path = os.path.join(folder_name, "processed_video.mp4")
csv_output_path = os.path.join(folder_name, "motion_data.csv")
graph_output_path = os.path.join(folder_name, "summary_report.png")

# ==========================================
# ⚙️ Config & Thresholds
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
SMOOTH_WINDOW = 7      
VISIBILITY_THRESHOLD = 0.6 
LEAN_THRESHOLD = 45    
MIN_FEET_RATIO = 0.5   
MAX_FEET_RATIO = 1.4   
BAD_POSTURE_DELAY = 3   
INCOMPLETE_STAND_DELAY = 15 

# --- Helper Functions ---
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# --- Setup Video ---
if VIDEO_SOURCE is None:
    cap = cv2.VideoCapture(0)
    print("Mode: WEBCAM")
else:
    if not os.path.exists(VIDEO_SOURCE): print(f"Error: File not found!"); exit()
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    print(f"Mode: VIDEO FILE")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

# --- Variables ---
counter = 0 
stage = None 
program_start = time.time()
angle_buffer = deque(maxlen=SMOOTH_WINDOW)
time_history = []
angle_history = []
rep_quality_history = [] 
current_rep_error = False 
bad_posture_counter = 0        
incomplete_stand_counter = 0   

print("System Ready. Run indefinitely until 'q' is pressed.")

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if VIDEO_SOURCE is None: frame = cv2.flip(frame, 1)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_angle = 0 
        feedback = "READY"
        feedback_color = (0, 255, 0) 

        if VIDEO_SOURCE is None: current_time_seconds = time.time() - program_start
        else: current_time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        try:
            landmarks = results.pose_landmarks.landmark
            vis_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
            vis_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
            
            if vis_hip < VISIBILITY_THRESHOLD or vis_knee < VISIBILITY_THRESHOLD:
                feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
            else:
                r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                raw_angle = calculate_angle(r_hip, r_knee, r_ankle)
                angle_buffer.append(raw_angle)
                current_angle = sum(angle_buffer) / len(angle_buffer)
                torso_lean = calculate_vertical_angle(r_shoulder, r_hip)
                shoulder_width = calculate_distance(l_shoulder, r_shoulder)
                feet_width = calculate_distance(l_ankle, r_ankle)
                stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                time_history.append(current_time_seconds)
                angle_history.append(current_angle)
                
                cv2.putText(image, str(int(current_angle)), tuple(np.multiply(r_knee, [frame_width, frame_height]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # --- SAFETY CHECK ---
                potential_bad_posture = False; temp_feedback = ""
                if stance_ratio < MIN_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                elif torso_lean > LEAN_THRESHOLD and (100 < current_angle < 160): potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                
                if potential_bad_posture: bad_posture_counter += 1
                else: bad_posture_counter = 0 
                
                potential_incomplete_stand = False
                if stage == 'up' and 140 < current_angle < 155: potential_incomplete_stand = True
                if potential_incomplete_stand: incomplete_stand_counter += 1
                else: incomplete_stand_counter = 0

                if bad_posture_counter > BAD_POSTURE_DELAY:
                    feedback = temp_feedback; feedback_color = (0, 0, 255); current_rep_error = True 
                elif incomplete_stand_counter > INCOMPLETE_STAND_DELAY:
                    feedback = "STAND UP FULLY!"; feedback_color = (0, 165, 255); current_rep_error = True
                else:
                    feedback = "GOOD FORM"; feedback_color = (0, 255, 0)

                # --- COUNTING (Unlimited Logic) ---
                if current_angle > 160: 
                    stage = "up"
                    if feedback == "GOOD FORM": # Reset error if good form
                        current_rep_error = False; bad_posture_counter = 0; incomplete_stand_counter = 0
                
                if current_angle < 85 and stage == 'up':
                    stage = "down"
                    counter += 1
                    if not current_rep_error: rep_quality_history.append(1) 
                    else: rep_quality_history.append(0) 
                    current_rep_error = False 
                
        except: pass
        
        # --- UI ---
        cv2.rectangle(image, (0,0), (640, 85), (245,117,16), -1)
        cv2.putText(image, 'REPS', (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, str(counter), (10,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        cv2.putText(image, 'FEEDBACK', (120,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, feedback, (120,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        
        current_acc = 0.0
        if len(rep_quality_history) > 0: current_acc = (sum(rep_quality_history) / len(rep_quality_history)) * 100
        cv2.putText(image, 'ACC', (400,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (400,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.putText(image, 'TIME', (520,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (505,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imshow('Sit-to-Stand Analysis', image)
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()

# --- Graph & Report ---
if len(time_history) > 0 and len(rep_quality_history) > 0:
    total_reps = len(rep_quality_history)
    correct_reps = sum(rep_quality_history)
    incorrect_reps = total_reps - correct_reps
    accuracy_percent = (correct_reps / total_reps) * 100 if total_reps > 0 else 0
    session_duration = time_history[-1] - time_history[0]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Graph 1: Movement
    ax1.plot(time_history, angle_history, label='Knee Angle', color='blue')
    ax1.axhline(y=160, color='g', linestyle='--', label='Stand')
    ax1.axhline(y=85, color='r', linestyle='--', label='Sit')
    ax1.set_title(f'Movement Analysis')
    ax1.set_ylabel('Knee Angle (deg)')
    ax1.grid(True); ax1.legend()
    
    # Graph 2: Summary
    labels = ['Correct', 'Incorrect']
    counts = [correct_reps, incorrect_reps]
    bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'], width=0.5)
    ax2.set_title('Performance Summary')
    ax2.set_ylabel('Count'); ax2.set_yticks(range(0, max(counts)+2))
    
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height()}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    summary_text = f"Total Reps: {total_reps}\nTotal Time: {session_duration:.2f} s\nAccuracy: {accuracy_percent:.1f}%"
    ax2.text(0.95, 0.95, summary_text, transform=ax2.transAxes, ha='right', va='top', fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(graph_output_path)
    plt.show()
    
    with open(csv_output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Knee_Angle'])
        for t, a in zip(time_history, angle_history): writer.writerow([t, a])

    print(f"✅ Saved to: {folder_name}")
else:
    print("No complete reps detected.")