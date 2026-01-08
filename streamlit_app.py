import os
import sys
import time

# -------------------------------------------------------------------------
# 🔧 FORCED CPU MODE
# -------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import av
import math
from collections import deque
import tempfile
import matplotlib.pyplot as plt

# ==========================================
# ⚙️ Configuration & Thresholds
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Smoothing & Logic Constants
SMOOTH_WINDOW = 7
VISIBILITY_THRESHOLD = 0.5
LEAN_THRESHOLD = 45
MIN_FEET_RATIO = 0.5
MAX_FEET_RATIO = 1.4
BAD_POSTURE_DELAY = 3
INCOMPLETE_STAND_DELAY = 15

# ==========================================
# 📐 Helper Functions
# ==========================================
def calculate_angle(a, b, c):
    """ Calculate angle between three points (a, b, c) """
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_distance(p1, p2):
    """ Calculate Euclidean distance """
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    """ Calculate angle relative to vertical axis """
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# ==========================================
# 🧠 Logic Class (Smart Leg + Original UI)
# ==========================================
class SitToStandLogic:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.counter = 0
        self.stage = None
        self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = [] 
        
        # State Flags
        self.current_rep_error = False
        self.bad_posture_counter = 0
        self.incomplete_stand_counter = 0
        self.current_side = "AUTO"

    def process_frame(self, image):
        # Initialize timer on first frame
        if self.start_time is None: self.start_time = time.time()
        
        # Resize for performance (Standard mobile width)
        target_w = 640
        h, w, c = image.shape
        scale = target_w / w
        new_h = int(h * scale)
        image = cv2.resize(image, (target_w, new_h))
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        
        # Defaults
        current_angle = 0
        feedback = "READY"
        feedback_color = (0, 255, 0) # Green
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # =========================================================
                # 🧠 SMART LEG SELECTION (Longest Thigh Logic)
                # =========================================================
                def get_raw(lm): return [lm.x, lm.y]
                
                # Get Landmarks for both sides
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                # Calculate Apparent Thigh Length (2D Projection)
                len_thigh_left = math.hypot(l_knee.x - l_hip.x, l_knee.y - l_hip.y)
                len_thigh_right = math.hypot(r_knee.x - r_hip.x, r_knee.y - r_hip.y)

                # Get Visibility
                vis_left = (l_hip.visibility + l_knee.visibility) / 2
                vis_right = (r_hip.visibility + r_knee.visibility) / 2

                # Scoring: Weight Length 80%, Visibility 20%
                score_left = (len_thigh_left * 3.0) + vis_left
                score_right = (len_thigh_right * 3.0) + vis_right

                # Select Best Side
                if score_left > score_right:
                    self.current_side = "LEFT"
                    hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
                    knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
                    ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
                    shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
                else:
                    self.current_side = "RIGHT"
                    hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
                    knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
                    ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value
                    shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
                # =========================================================

                # Check Visibility of Selected Leg
                selected_knee_vis = landmarks[knee_idx].visibility
                
                if selected_knee_vis < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    # Extract Keypoints
                    hip_raw = get_raw(landmarks[hip_idx])
                    knee_raw = get_raw(landmarks[knee_idx])
                    ankle_raw = get_raw(landmarks[ankle_idx])
                    shoulder_raw = get_raw(landmarks[shoulder_idx])
                    
                    # For Stance Check (Need Both)
                    r_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                    l_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                    r_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                    l_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                    # Calculate Angles
                    raw_angle = calculate_angle(hip_raw, knee_raw, ankle_raw)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    torso_lean = calculate_vertical_angle(shoulder_raw, hip_raw)
                    shoulder_width = calculate_distance(l_shoulder_raw, r_shoulder_raw)
                    feet_width = calculate_distance(l_ankle_raw, r_ankle_raw)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    # Draw Angle
                    knee_px = tuple(np.multiply(knee_raw, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # --- SAFETY CHECK ---
                    potential_bad_posture = False; temp_feedback = ""
                    
                    if stance_ratio < MIN_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                    elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                    elif torso_lean > LEAN_THRESHOLD and (100 < current_angle < 160): 
                        potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                    
                    if potential_bad_posture: self.bad_posture_counter += 1
                    else: self.bad_posture_counter = 0 
                    
                    potential_inc = False
                    if self.stage == 'up' and 140 < current_angle < 155: potential_inc = True
                    if potential_inc: self.incomplete_stand_counter += 1
                    else: self.incomplete_stand_counter = 0

                    # Decision
                    if self.bad_posture_counter > BAD_POSTURE_DELAY:
                        feedback = temp_feedback; feedback_color = (0, 0, 255); self.current_rep_error = True 
                    elif self.incomplete_stand_counter > INCOMPLETE_STAND_DELAY:
                        feedback = "STAND UP FULLY!"; feedback_color = (0, 165, 255); self.current_rep_error = True
                    else:
                        feedback = "GOOD FORM"; feedback_color = (0, 255, 0)

                    # --- REPETITION COUNTING ---
                    if current_angle > 160: 
                        self.stage = "up"
                        if feedback == "GOOD FORM": 
                            self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
                    
                    if current_angle < 85 and self.stage == 'up':
                        self.stage = "down"
                        self.counter += 1
                        if not self.current_rep_error: self.rep_quality_history.append(1) 
                        else: self.rep_quality_history.append(0) 
                        self.current_rep_error = False 

            except Exception as e: pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ------------------------------------------------------------------
        # 🎨 INTERFACE RESTORED (Original Top Bar Style)
        # ------------------------------------------------------------------
        cv2.rectangle(image, (0,0), (target_w, 85), (245,117,16), -1)
        
        x_rep = 15; x_feed = int(target_w * 0.2); x_acc = int(target_w * 0.65); x_time = int(target_w * 0.85)

        cv2.putText(image, 'REPS', (x_rep,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, str(self.counter), (x_rep-5,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        cv2.putText(image, 'FEEDBACK', (x_feed,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, feedback, (x_feed,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        
        cv2.putText(image, 'ACC', (x_acc,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (x_acc,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.putText(image, 'TIME', (x_time,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (x_time,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        # Debug: Show Active Leg
        cv2.putText(image, f"Active: {self.current_side}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return image, current_angle, current_time_seconds

# ==========================================
# 🌐 Streamlit Interface (Main App)
# ==========================================
st.set_page_config(page_title="STS Analyzer", layout="wide")
st.title("🩺 AI-Based STS Biomechanics Analyzer")
st.markdown("**Web Version:** Runs on iPad/iPhone/Android/PC")

# 1. Initialize Session State for Webcam Data
if "webcam_results" not in st.session_state:
    st.session_state["webcam_results"] = None

mode = st.radio("Select Input Source:", ("Webcam (Live)", "Video File"))

if mode == "Webcam (Live)":
    class VideoProcessor(VideoTransformerBase):
        def __init__(self): 
            self.logic = SitToStandLogic()
            self.angle_history = []
            self.time_history = []
        
        def recv(self, frame):
            try:
                time.sleep(0.01) # Yield CPU
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1) # Mirror
                
                # Process
                processed_img, angle, timestamp = self.logic.process_frame(img)
                
                # Record Data
                self.angle_history.append(angle)
                self.time_history.append(timestamp)
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as e:
                return frame
        
        def get_stats(self):
            return {
                "rep_quality_history": self.logic.rep_quality_history,
                "angle_history": self.angle_history,
                "time_history": self.time_history
            }

    st.info("💡 Instructions: Click 'START'. When finished, click 'STOP' to see results. If WiFi fails, try using Mobile Hotspot.")
    
    # Auto-TURN Config
    ctx = webrtc_streamer(
        key="sts-webcam-final-v16",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {"width": 320, "height": 240, "frameRate": 15},
            "audio": False
        },
        async_processing=True,
    )

    # 📊 Logic to capture data AFTER stop
    if ctx.video_processor:
        # Save data while running
        st.session_state["webcam_results"] = ctx.video_processor.get_stats()

    # If stream stopped AND we have data -> Show Graphs
    if not ctx.state.playing and st.session_state["webcam_results"]:
        data = st.session_state["webcam_results"]
        rep_history = data["rep_quality_history"]
        angle_hist = data["angle_history"]
        time_hist = data["time_history"]
        
        if len(rep_history) > 0 or len(angle_hist) > 0:
            st.divider()
            st.subheader("📊 Session Summary (Webcam)")
            
            total_reps = len(rep_history)
            correct_reps = sum(rep_history)
            accuracy = (correct_reps/total_reps*100) if total_reps > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reps", total_reps)
            col2.metric("Good Form", correct_reps)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Graph 1
            ax1.plot(time_hist, angle_hist, label='Knee Angle', color='blue')
            ax1.axhline(y=160, color='g', linestyle='--', label='Stand (160°)')
            ax1.axhline(y=85, color='r', linestyle='--', label='Sit (85°)')
            ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
            
            # Graph 2
            labels = ['Correct', 'Incorrect']
            counts = [correct_reps, total_reps - correct_reps]
            bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
            ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
            for bar in bars: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Clear state button
            if st.button("Start New Session"):
                st.session_state["webcam_results"] = None
                st.rerun()

elif mode == "Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            # 1. Setup Video Writer
            logic = SitToStandLogic()
            angle_data = []; time_data = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create Temp Output
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            
            # Match dimensions
            target_w = 640
            original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = target_w / original_w
            target_h = int(original_h * scale)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(temp_output.name, fourcc, fps, (target_w, target_h))

            # 2. UI Elements
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.info("⏳ Processing video... Please wait.")
            
            # 3. Process Loop
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_img, angle, timestamp = logic.process_frame(frame)
                out.write(processed_img)
                
                angle_data.append(angle)
                time_data.append(timestamp)
                
                frame_count += 1
                if total_frames > 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}")

            # 4. Cleanup & Converter
            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
            
            # ⚙️ CONVERT TO H.264 (Fixes black screen in browser)
            converted_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            os.system(f"ffmpeg -y -i {temp_output.name} -vcodec libx264 {converted_output.name}")

            # ==========================================
            # 📊 Show Results
            # ==========================================
            st.success("✅ Analysis Complete!")
            
            st.subheader("🎬 Analyzed Video")
            st.video(converted_output.name)
            
            st.divider()
            st.subheader("📊 Summary Report")
            
            total_reps = len(logic.rep_quality_history)
            correct_reps = sum(logic.rep_quality_history)
            accuracy = (correct_reps/total_reps*100) if total_reps > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reps", total_reps)
            col2.metric("Good Form", correct_reps)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(time_data, angle_data, label='Knee Angle', color='blue')
            ax1.axhline(y=160, color='g', linestyle='--', label='Stand (160°)')
            ax1.axhline(y=85, color='r', linestyle='--', label='Sit (85°)')
            ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
            labels = ['Correct', 'Incorrect']; counts = [correct_reps, total_reps - correct_reps]
            bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
            ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
            for bar in bars: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
            st.pyplot(fig)