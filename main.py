"""
Sit-to-Stand (STS) Biomechanics Analyzer [Desktop Version]

Description:
    A desktop application for analyzing the Sit-to-Stand test using Computer Vision.
    It utilizes MediaPipe Pose for skeleton tracking and calculates 3D biomechanical angles
    to assess performance, count repetitions, and provide real-time feedback on posture.

Features:
    - Real-time Skeleton Tracking (MediaPipe)
    - 3D Biomechanical Angle Calculation (Knee, Hip, Ankle)
    - Automatic Repetition Counting (Sit/Stand State Machine)
    - Real-time Feedback (Posture, Speed, Form)
    - Post-session Analysis Dashboard (Matplotlib)
    - CSV Data Export
    - Smart Window Resizing for Desktop UI
"""

import os
import sys
import time
import math
import csv
import threading
from collections import deque
from datetime import datetime

# Data Science & CV
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

# System & GUI
import psutil
import gc
import tkinter as tk
from tkinter import filedialog, messagebox

# ==========================================
# 🛡️ SYSTEM MONITORING (Background Thread)
# ==========================================
def get_current_memory_mb():
    """Returns current process memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def start_ram_monitor():
    """Starts a daemon thread to monitor RAM usage and trigger GC if critical."""
    def monitor_loop():
        while True:
            mem = get_current_memory_mb()
            # Safety threshold for desktop application (2.5 GB)
            if mem > 2500: 
                print(f"⚠️ High RAM Usage Detected ({mem:.1f} MB)! Triggering Garbage Collection...")
                gc.collect()
            time.sleep(5)
    
    t = threading.Thread(target=monitor_loop, name="RamMonitor", daemon=True)
    t.start()

# Initialize Memory Guard
start_ram_monitor()

# ==========================================
# ⚙️ CONFIGURATION & CONSTANTS
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Signal Smoothing
SMOOTH_WINDOW = 7           # Window size for moving average filter

# Detection Thresholds
VISIBILITY_THRESHOLD = 0.5  # Minimum landmark visibility score

# Biomechanics Thresholds
LEAN_THRESHOLD = 45         # Max forward torso lean (degrees)
MIN_FEET_RATIO = 0.5        # Min stance width (shoulder width ratio)
MAX_FEET_RATIO = 1.4        # Max stance width

# Debounce Delays (Frame Counts)
BAD_POSTURE_DELAY = 3       # Frames to wait before triggering posture warning
INCOMPLETE_STAND_DELAY = 15 # Frames to wait before warning about incomplete stand

# ==========================================
# 📐 MATHEMATICAL HELPER FUNCTIONS
# ==========================================
def calculate_angle(a, b, c):
    """Calculates the 2D angle between three points (a-b-c) on a plane."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_angle_3d(a, b, c):
    """
    Calculates the 3D angle using vector dot product.
    Essential for accurate knee angle measurement invariant to camera perspective.
    """
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0: return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    """Calculates the angle of a segment (a-b) relative to the vertical axis."""
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# ==========================================
# 🧠 CORE LOGIC ENGINE
# ==========================================
class SitToStandLogic:
    def __init__(self):
        # Initialize MediaPipe Pose with standard complexity for Desktop performance
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7, 
            model_complexity=1
        )
        # State variables
        self.counter = 0
        self.stage = None
        self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = []  # 1 = Good, 0 = Bad
        
        # Error tracking
        self.current_rep_error = False
        self.bad_posture_counter = 0
        self.incomplete_stand_counter = 0
        self.current_side = "AUTO"

    def close(self):
        """Clean up MediaPipe resources."""
        if self.pose: self.pose.close()

    def process_frame(self, image):
        """
        Main pipeline: Image -> Inference -> Logic -> UI Overlay.
        Returns: Processed Image, Current Angle, Timestamp, Rep History
        """
        if self.start_time is None: self.start_time = time.time()
        
        # 1. Standardization: Resize image to fixed width for consistent processing
        target_w = 1280 
        h, w, c = image.shape
        if w > target_w:
            scale = target_w / w
            new_h = int(h * scale)
            image = cv2.resize(image, (target_w, new_h))
        else: target_w = w; new_h = h
        
        # 2. Inference
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        current_angle = 0
        feedback = "READY"
        feedback_color = (0, 255, 0) # Green
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks and results.pose_world_landmarks:
            try:
                # Extract Landmarks
                landmarks = results.pose_landmarks.landmark      # 2D normalized coords
                world_landmarks = results.pose_world_landmarks.landmark # 3D real-world meters

                def get_2d(lm): return [lm.x, lm.y]
                def get_3d(lm): return [lm.x, lm.y, lm.z]
                
                # --- Auto Side Detection Logic ---
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                # Determine side based on limb visibility and size
                len_left = math.hypot(l_knee.x - l_hip.x, l_knee.y - l_hip.y)
                len_right = math.hypot(r_knee.x - r_hip.x, r_knee.y - r_hip.y)
                score_left = (len_left * 3.0) + ((l_hip.visibility + l_knee.visibility)/2)
                score_right = (len_right * 3.0) + ((r_hip.visibility + r_knee.visibility)/2)

                if score_left > score_right:
                    self.current_side = "LEFT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 23, 25, 27, 11
                else:
                    self.current_side = "RIGHT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 24, 26, 28, 12

                selected_knee_vis = landmarks[knee_idx].visibility
                
                if selected_knee_vis < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    # --- Data Extraction ---
                    hip_3d = get_3d(world_landmarks[hip_idx])
                    knee_3d = get_3d(world_landmarks[knee_idx])
                    ankle_3d = get_3d(world_landmarks[ankle_idx])

                    hip_2d = get_2d(landmarks[hip_idx])
                    knee_2d = get_2d(landmarks[knee_idx])
                    shoulder_2d = get_2d(landmarks[shoulder_idx])
                    
                    # For stance analysis (using both legs)
                    l_shoulder_2d = get_2d(landmarks[11])
                    r_shoulder_2d = get_2d(landmarks[12])
                    l_ankle_2d = get_2d(landmarks[27])
                    r_ankle_2d = get_2d(landmarks[28])

                    # --- Biomechanical Calculations ---
                    # 1. Main Joint Angle (3D for accuracy)
                    raw_angle = calculate_angle_3d(hip_3d, knee_3d, ankle_3d)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    # 2. Auxiliary Metrics (2D for visual feedback)
                    angle_2d = calculate_angle(hip_2d, knee_2d, get_2d(landmarks[ankle_idx]))
                    torso_lean = calculate_vertical_angle(shoulder_2d, hip_2d)
                    shoulder_width = calculate_distance(l_shoulder_2d, r_shoulder_2d)
                    feet_width = calculate_distance(l_ankle_2d, r_ankle_2d)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    # --- Visualization ---
                    knee_px = tuple(np.multiply(knee_2d, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # --- Feedback & Safety Logic ---
                    potential_bad_posture = False; temp_feedback = ""
                    
                    # Check Stance Width
                    if stance_ratio < MIN_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                    elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                    # Check Torso Lean
                    elif torso_lean > LEAN_THRESHOLD and (100 < angle_2d < 160): 
                        potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                    
                    # Debounce Bad Posture
                    if potential_bad_posture: self.bad_posture_counter += 1
                    else: self.bad_posture_counter = 0 
                    
                    # Check Incomplete Stand
                    potential_inc = False
                    if self.stage == 'up' and 140 < current_angle < 155: potential_inc = True
                    if potential_inc: self.incomplete_stand_counter += 1
                    else: self.incomplete_stand_counter = 0

                    # Decision
                    if self.bad_posture_counter > BAD_POSTURE_DELAY:
                        feedback = temp_feedback; feedback_color = (0, 0, 255); self.current_rep_error = True 
                    elif self.incomplete_stand_counter > INCOMPLETE_STAND_DELAY:
                        feedback = "STAND UP FULLY!"; feedback_color = (0, 165, 255); self.current_rep_error = True
                    else: feedback = "GOOD FORM"; feedback_color = (0, 255, 0)

                    # --- Repetition Counting State Machine ---
                    if current_angle > 165: # Standing Phase
                        self.stage = "up"
                        # Allow error recovery if good form is re-established in standing position
                        if feedback == "GOOD FORM": 
                            self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
                    
                    if current_angle < 100 and self.stage == 'up': # Sitting Phase
                        self.stage = "down"; self.counter += 1
                        
                        # Log Quality
                        if not self.current_rep_error: self.rep_quality_history.append(1) 
                        else: self.rep_quality_history.append(0) 
                        
                        # Reset for next rep
                        self.current_rep_error = False 

            except Exception: pass
            
            # Draw Skeleton Overlay
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 3. UI Dashboard Overlay
        cv2.rectangle(image, (0,0), (target_w, 85), (245,117,16), -1)
        x_rep = 20; x_feed = int(target_w * 0.25); x_acc = int(target_w * 0.65); x_time = int(target_w * 0.85)
        font_scale = 0.8 if target_w > 1000 else 0.6; font_thick = 2

        # Draw Stats
        cv2.putText(image, 'REPS', (x_rep,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, str(self.counter), (x_rep-5,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        cv2.putText(image, 'FEEDBACK', (x_feed,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, feedback, (x_feed,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: 
            current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        
        cv2.putText(image, 'ACC', (x_acc,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (x_acc,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.putText(image, 'TIME', (x_time,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (x_time,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        cv2.putText(image, f"Active: {self.current_side}", (20, target_w - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        exit_text = "Press 'q' to Finish"
        text_size = cv2.getTextSize(exit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = target_w - text_size[0] - 20
        cv2.putText(image, exit_text, (text_x, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return image, current_angle, current_time_seconds, self.rep_quality_history

# ==========================================
# 🖥️ UI HELPER: SMART RESIZE
# ==========================================
def calculate_optimal_size(original_w, original_h):
    """Calculates video dimensions to fit approx 85% of the user's screen."""
    root = tk.Tk()
    root.withdraw()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()

    target_w = screen_w * 0.85
    target_h = screen_h * 0.85

    ratio_w = target_w / original_w
    ratio_h = target_h / original_h
    scale = min(ratio_w, ratio_h)
    
    return int(original_w * scale), int(original_h * scale)

# ==========================================
# 📂 UI HELPER: FILE SELECTOR
# ==========================================
def get_video_source():
    """Opens a dialog to select Webcam or File input."""
    root = tk.Tk()
    root.withdraw()
    
    # Prompt user
    use_webcam = messagebox.askyesno(
        "Select Input Source", 
        "Do you want to use the WEBCAM?\n\n(Yes = Webcam, No = Video File)"
    )
    
    if use_webcam:
        root.destroy()
        return None 
    else:
        file_path = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
        )
        root.destroy()
        if not file_path: exit()
        return file_path

# ==========================================
# 🚀 MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # --- 1. Initialization ---
    VIDEO_SOURCE = get_video_source()
    
    # Create Output Directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if VIDEO_SOURCE is None:
        folder_name = f"SitToStand_Webcam_{timestamp}"
        cap = cv2.VideoCapture(0)
    else:
        base_name = os.path.basename(VIDEO_SOURCE).split('.')[0]
        folder_name = f"SitToStand_{base_name}_{timestamp}"
        if not os.path.exists(VIDEO_SOURCE): exit()
        cap = cv2.VideoCapture(VIDEO_SOURCE)

    os.makedirs(folder_name, exist_ok=True)
    print(f"📂 Output Folder Created: {folder_name}")
    
    # Output Paths
    video_output_path = os.path.join(folder_name, "processed_video.mp4")
    csv_output_path = os.path.join(folder_name, "motion_data.csv")
    graph_output_path = os.path.join(folder_name, "summary_report.png")

    # --- 2. Setup Logic & Video Writer ---
    logic = SitToStandLogic() 
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # Read first frame to setup dimensions
    ret, frame = cap.read()
    if not ret: exit()
    if VIDEO_SOURCE is not None: cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Standardize processing width
    processed_w = 1280
    h, w, c = frame.shape
    scale_factor = processed_w / w
    processed_h = int(h * scale_factor)
    
    # Calculate display size
    display_w, display_h = calculate_optimal_size(processed_w, processed_h)
    
    # Init Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (processed_w, processed_h))

    # Data Logging Containers
    all_time_history = []
    all_angle_history = []
    final_rep_history = []

    print("🚀 System Ready. Press 'q' to STOP and view SUMMARY.")

    # --- 3. Main Processing Loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Mirror view for Webcam
        if VIDEO_SOURCE is None: frame = cv2.flip(frame, 1)

        # Process Frame
        processed_img, angle, timestamp, rep_history = logic.process_frame(frame)
        
        # Log Data
        all_time_history.append(timestamp)
        all_angle_history.append(angle)
        final_rep_history = rep_history[:] 

        # Save & Display
        out.write(processed_img)
        display_img = cv2.resize(processed_img, (display_w, display_h))
        cv2.imshow('Sit-to-Stand Analyzer (Press q to Finish)', display_img)

        # Exit Condition
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # --- 4. Cleanup & Report Generation ---
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logic.close()

    # Generate Report if data exists (Safe for 0 reps)
    if len(all_time_history) > 0:
        # Save Raw CSV
        with open(csv_output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Knee_Angle'])
            for t, a in zip(all_time_history, all_angle_history): writer.writerow([t, a])
        print(f"📝 CSV saved.")

        # Calculate Statistics
        total_reps = len(final_rep_history)
        correct_reps = sum(final_rep_history)
        incorrect_reps = total_reps - correct_reps
        accuracy_percent = (correct_reps / total_reps) * 100 if total_reps > 0 else 0
        session_duration = all_time_history[-1] - all_time_history[0] if len(all_time_history) > 1 else 0

        # Create Dashboard Visualization
        plt.ioff()
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle('Sit-to-Stand Session Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Angle vs Time
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(all_time_history, all_angle_history, label='Knee Angle', color='blue', linewidth=2)
        ax1.axhline(y=165, color='g', linestyle='--', label='Stand (165°)', alpha=0.7)
        ax1.axhline(y=100, color='r', linestyle='--', label='Sit (100°)', alpha=0.7)
        ax1.set_title('Knee Angle Movement Analysis')
        ax1.set_ylabel('Angle (deg)')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Plot 2: Bar Chart
        ax2 = fig.add_subplot(2, 2, 3)
        labels = ['Correct', 'Incorrect']
        counts = [correct_reps, incorrect_reps]
        bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'], width=0.6)
        ax2.set_title('Repetition Quality')
        ax2.set_ylabel('Count')
        y_max = max(counts) if total_reps > 0 else 5
        ax2.set_yticks(range(0, y_max + 2))
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontweight='bold')

        # Plot 3: Text Summary
        ax3 = fig.add_subplot(2, 2, 4)
        ax3.axis('off')
        
        summary_text = (
            f"SESSION RESULTS\n"
            f"----------------\n"
            f"Total Reps:   {total_reps}\n"
            f"Good Form:    {correct_reps}\n"
            f"Accuracy:     {accuracy_percent:.1f}%\n"
            f"Duration:     {session_duration:.1f} sec\n"
            f"Date:         {datetime.now().strftime('%Y-%m-%d')}"
        )
        
        ax3.text(0.5, 0.5, summary_text, 
                 ha='center', va='center', fontsize=14, 
                 bbox=dict(boxstyle="round,pad=1", facecolor='#f0f0f0', edgecolor='gray'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save and Show
        plt.savefig(graph_output_path)
        print(f"📊 Graph saved to: {graph_output_path}")
        
        print("📈 Opening Summary Dashboard... (Close the graph window to finish)")
        plt.show()
        
    print(f"✅ All Done! Results saved in: {folder_name}")