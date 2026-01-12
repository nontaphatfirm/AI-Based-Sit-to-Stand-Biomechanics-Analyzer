"""
Sit-to-Stand (STS) Biomechanics Analyzer [Web Version]

Description:
    A Streamlit-based web application for analyzing the Sit-to-Stand test using Computer Vision.
    It leverages MediaPipe Pose for real-time skeleton tracking and biomechanical analysis via 
    WebRTC (for webcam) and file upload processing.

Features:
    - Real-time Webcam Analysis via WebRTC
    - Video File Upload & Processing
    - 3D Biomechanical Angle Calculation (Knee, Hip, Ankle)
    - Automatic Repetition Counting & Form Feedback
    - Post-session Analytics Dashboard (Matplotlib)
    - Memory Management & Auto-Safety Guards (No manual reset required)
"""

import os
import sys
import time
import gc
import uuid
import hashlib
import psutil
import ctypes
import threading
import io
from collections import deque
import tempfile
import math

# Third-party Libraries
import cv2
import mediapipe as mp
import numpy as np
import av
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# -------------------------------------------------------------------------
# 🔧 SYSTEM CONFIGURATION: FORCED CPU MODE
# -------------------------------------------------------------------------
# Force TensorFlow/MediaPipe to run on CPU to avoid GPU conflicts in cloud environments
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
os.environ["MESA_LOADER_DRIVER_OVERRIDE"] = "llvmpipe"

# ==========================================
# 🧹 MEMORY MANAGEMENT & CLEANUP UTILS
# ==========================================
def nuclear_cleanup():
    """
    Aggressively releases memory back to the OS.
    Crucial for long-running Streamlit sessions to prevent memory leaks.
    """
    # Run Garbage Collection multiple times to clear cyclical references
    for _ in range(3):
        gc.collect()
    
    # Force libc to release memory (Linux specific, helpful for Streamlit Cloud)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

# Perform cleanup on initial script load
nuclear_cleanup()

def get_current_memory_mb():
    """Returns current process memory usage in MB."""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def start_ram_monitor():
    """Starts a background thread to log RAM usage to console."""
    for t in threading.enumerate():
        if t.name == "RamMonitor":
            return
    def monitor_loop():
        while True:
            mem = get_current_memory_mb()
            # Log backend metrics (visible in cloud logs)
            print(f"📈 [System Monitor] RAM Usage: {mem:.1f} MB", flush=True)
            time.sleep(3)
            
    t = threading.Thread(target=monitor_loop, name="RamMonitor", daemon=True)
    t.start()
    print("✅ RAM Monitor Thread Started!", flush=True)

# Initialize RAM Monitor
start_ram_monitor()

def check_memory_safe(limit_mb=1500):
    """Checks if memory usage is within safe limits."""
    current_mem_mb = get_current_memory_mb()
    if current_mem_mb > limit_mb:
        return False, current_mem_mb
    return True, current_mem_mb

# ==========================================
# ⚙️ APP CONSTANTS & THRESHOLDS
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Smoothing & Filtering
SMOOTH_WINDOW = 7           # Moving average window size

# Detection Quality
VISIBILITY_THRESHOLD = 0.5  # Min landmark visibility

# Biomechanics
LEAN_THRESHOLD = 45         # Max torso forward lean (deg)
MIN_FEET_RATIO = 0.5        # Min stance width (shoulder ratio)
MAX_FEET_RATIO = 1.4        # Max stance width

# Debounce Config (Frames)
BAD_POSTURE_DELAY = 3       
INCOMPLETE_STAND_DELAY = 15 

# ==========================================
# 📐 GEOMETRY HELPER FUNCTIONS
# ==========================================
def calculate_angle(a, b, c):
    """Calculates 2D angle between points a-b-c."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_angle_3d(a, b, c):
    """
    Calculates 3D angle using vector dot product.
    Provides robust knee angle measurement invariant to camera perspective.
    """
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    
    # Avoid division by zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    """Calculates 2D Euclidean distance."""
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    """Calculates angle relative to vertical axis (for torso lean)."""
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# ==========================================
# 🧠 CORE LOGIC ENGINE
# ==========================================
class SitToStandLogic:
    """
    Encapsulates the core business logic for STS analysis:
    - MediaPipe Pose Inference
    - Biomechanical Calculations
    - Repetition Counting State Machine
    - Real-time Feedback Generation
    """
    def __init__(self):
        # Initialize Pose model (Complexity 1 for balance speed/accuracy)
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.7, 
            min_tracking_confidence=0.7, 
            model_complexity=1
        )
        self.counter = 0
        self.stage = None
        self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = [] 
        self.current_rep_error = False
        self.bad_posture_counter = 0
        self.incomplete_stand_counter = 0
        self.current_side = "AUTO"

    def close(self):
        """Explicitly release MediaPipe resources."""
        if self.pose:
            self.pose.close()
            self.pose = None

    def __del__(self):
        self.close()

    def process_frame(self, image):
        """
        Main processing pipeline for a single frame.
        """
        if self.start_time is None: self.start_time = time.time()
        
        # 1. Resize for consistent processing speed
        target_w = 1280 
        h, w, c = image.shape
        if w > target_w:
            scale = target_w / w
            new_h = int(h * scale)
            image = cv2.resize(image, (target_w, new_h))
        else: target_w = w; new_h = h
        
        # 2. Run Inference
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        
        current_angle = 0
        feedback = "READY"
        feedback_color = (0, 255, 0)
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks and results.pose_world_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark
                
                def get_2d(lm): return [lm.x, lm.y]
                def get_3d(lm): return [lm.x, lm.y, lm.z]
                
                # --- Auto Side Detection ---
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                # Determine active side based on visibility and size
                len_left = math.hypot(l_knee.x - l_hip.x, l_knee.y - l_hip.y)
                len_right = math.hypot(r_knee.x - r_hip.x, r_knee.y - r_hip.y)
                vis_left = (l_hip.visibility + l_knee.visibility) / 2
                vis_right = (r_hip.visibility + r_knee.visibility) / 2
                
                # Heuristic score
                score_left = (len_left * 3.0) + vis_left
                score_right = (len_right * 3.0) + vis_right

                if score_left > score_right:
                    self.current_side = "LEFT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 23, 25, 27, 11
                else:
                    self.current_side = "RIGHT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 24, 26, 28, 12

                # --- Biomechanical Analysis ---
                selected_knee_vis = landmarks[knee_idx].visibility
                if selected_knee_vis < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    # Extract 3D Coordinates
                    hip_3d = get_3d(world_landmarks[hip_idx])
                    knee_3d = get_3d(world_landmarks[knee_idx])
                    ankle_3d = get_3d(world_landmarks[ankle_idx])

                    # Extract 2D Coordinates
                    hip_2d = get_2d(landmarks[hip_idx])
                    knee_2d = get_2d(landmarks[knee_idx])
                    shoulder_2d = get_2d(landmarks[shoulder_idx])
                    l_shoulder_2d = get_2d(landmarks[11])
                    r_shoulder_2d = get_2d(landmarks[12])
                    l_ankle_2d = get_2d(landmarks[27])
                    r_ankle_2d = get_2d(landmarks[28])

                    # Calculate Metrics
                    raw_angle = calculate_angle_3d(hip_3d, knee_3d, ankle_3d)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    angle_2d = calculate_angle(hip_2d, knee_2d, get_2d(landmarks[ankle_idx]))
                    torso_lean = calculate_vertical_angle(shoulder_2d, hip_2d)
                    shoulder_width = calculate_distance(l_shoulder_2d, r_shoulder_2d)
                    feet_width = calculate_distance(l_ankle_2d, r_ankle_2d)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    # Draw Angle Text
                    knee_px = tuple(np.multiply(knee_2d, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    # --- Feedback Logic ---
                    potential_bad_posture = False; temp_feedback = ""
                    
                    # Stance Check
                    if stance_ratio < MIN_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                    elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: 
                        potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                    # Lean Check
                    elif torso_lean > LEAN_THRESHOLD and (100 < angle_2d < 160): 
                        potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                    
                    # Debounce
                    if potential_bad_posture: self.bad_posture_counter += 1
                    else: self.bad_posture_counter = 0 
                    
                    # Incomplete Stand Check
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

                    # --- State Machine (Counting) ---
                    if current_angle > 165: # Stand Threshold
                        self.stage = "up"
                        if feedback == "GOOD FORM": 
                            self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
                    
                    if current_angle < 100 and self.stage == 'up': # Sit Threshold
                        self.stage = "down"; self.counter += 1
                        # Log Quality
                        if not self.current_rep_error: self.rep_quality_history.append(1) 
                        else: self.rep_quality_history.append(0) 
                        self.current_rep_error = False 

            except Exception: pass
            
            # Draw Skeleton Overlay
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # --- UI Overlay ---
        cv2.rectangle(image, (0,0), (target_w, 85), (245,117,16), -1)
        x_rep = 20; x_feed = int(target_w * 0.25); x_acc = int(target_w * 0.65); x_time = int(target_w * 0.85)
        font_scale = 0.8 if target_w > 1000 else 0.6; font_thick = 2

        cv2.putText(image, 'REPS', (x_rep,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, str(self.counter), (x_rep-5,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*2, (255,255,255), font_thick)
        cv2.putText(image, 'FEEDBACK', (x_feed,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, feedback, (x_feed,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, font_thick)
        
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: 
            current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        
        cv2.putText(image, 'ACC', (x_acc,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (x_acc,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.5, (255,255,255), font_thick)
        
        cv2.putText(image, 'TIME', (x_time,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (x_time,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.2, (255,255,255), font_thick)
        
        cv2.putText(image, f"Active: {self.current_side}", (20, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return image, current_angle, current_time_seconds

# ==========================================
# 🌐 STREAMLIT UI SETUP
# ==========================================
st.set_page_config(page_title="STS Analyzer", layout="wide")

# No Sidebar Buttons as per requirement (Auto-Nuke is active)

st.title("🩺 AI-Based STS Biomechanics Analyzer")
st.markdown("**Web Version:** Optimized for iPad/iPhone/Android/PC")

# --- Session State Initialization ---
if "user_session_id" not in st.session_state:
    st.session_state["user_session_id"] = str(uuid.uuid4())[:8]

if "webrtc_key" not in st.session_state:
    st.session_state["webrtc_key"] = f"sts-v1-{uuid.uuid4()}"

if "webcam_results" not in st.session_state: st.session_state["webcam_results"] = None

# 🚨 Initial Safety Check (Auto-Reset if RAM is Critical)
safe, mem_usage = check_memory_safe(1500)
if not safe:
    st.error(f"⚠️ **System Reset triggered due to High Memory ({mem_usage:.1f} MB)**")
    st.warning("🔄 Cleaning up resources... Please refresh manually if needed.")
    st.cache_resource.clear()
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    nuclear_cleanup()
    st.stop()

# --- Mode Selection ---
mode = st.radio("Select Input Source:", ("Webcam (Live)", "Video File"))

# User Guide Expander
with st.expander("ℹ️ User Guide & Camera Setup (Click to open)", expanded=False):
    st.markdown(
        """
        ### 📸 Optimal Camera Positioning
        Our AI uses **3D Motion Analysis**, allowing it to track you from various angles. However, for the best results:
        
        1.  **📐 The "Sweet Spot" (45°):** Stand diagonally (approx. 45°) to the camera.
            * *Why?* This allows the AI to accurately measure **BOTH** your **Knee Angle** (for counting) and **Stance Width** (for posture check).
        2.  **📏 Full Body:** Ensure your **entire body** is visible from **Head to Toe** at all times.
        3.  **💡 Lighting:** Use a well-lit room and avoid wearing clothes that blend into the background.
        """
    )

# ==========================================
# 📹 MODE 1: WEBCAM (LIVE)
# ==========================================
if mode == "Webcam (Live)":
    class VideoProcessor(VideoTransformerBase):
        """
        Streamlit WebRTC Processor.
        Handles frame-by-frame processing for live video streams.
        """
        def __init__(self): 
            self.logic = SitToStandLogic()
            self.angle_history = []
            self.time_history = []
            self.frame_count = 0
        
        def close(self):
            if self.logic: self.logic.close()

        def recv(self, frame):
            try:
                # 🛡️ Safety Cut: Check memory every 10 frames
                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    safe, _ = check_memory_safe(1500)
                    if not safe:
                        print("🔥 RAM Limit Hit in Webcam! Stopping...")
                        raise Exception("Memory Limit Exceeded")

                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1) # Mirror effect for webcam
                
                # Processing
                processed_img, angle, timestamp = self.logic.process_frame(img)
                
                # Store Data
                self.angle_history.append(angle)
                self.time_history.append(timestamp)
                
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as e:
                raise e 
        
        def get_stats(self):
            return {
                "rep_quality_history": self.logic.rep_quality_history,
                "angle_history": self.angle_history,
                "time_history": self.time_history
            }

    st.info("💡 Instructions: Click 'START'. When finished, click 'STOP' to see results.")
    
    # Dynamic Key forces new component creation on reset (prevents thread hanging)
    ctx = webrtc_streamer(
        key=st.session_state["webrtc_key"], 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720, "frameRate": 30}, "audio": False},
        async_processing=True,
    )

    # Capture stats during streaming
    if ctx.video_processor:
        st.session_state["webcam_results"] = ctx.video_processor.get_stats()

    # Show Summary when stopped
    if not ctx.state.playing and st.session_state["webcam_results"]:
        data = st.session_state["webcam_results"]
        rep_history = data["rep_quality_history"]
        
        if len(data["time_history"]) > 0: # Ensure data exists
            st.divider()
            st.subheader("📊 Session Summary")
            
            total_reps = len(rep_history)
            correct_reps = sum(rep_history)
            accuracy = (correct_reps/total_reps*100) if total_reps > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reps", total_reps)
            col2.metric("Good Form", correct_reps)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
            
            # --- Visualization ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Angle Graph
            ax1.plot(data["time_history"], data["angle_history"], label='Knee Angle', color='blue')
            ax1.axhline(y=165, color='g', linestyle='--', label='Stand (165°)')
            ax1.axhline(y=100, color='r', linestyle='--', label='Sit (100°)')
            ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
            
            # Bar Chart
            labels = ['Correct', 'Incorrect']
            counts = [correct_reps, total_reps - correct_reps]
            bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
            ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
            
            # Annotate bars
            for bar in bars: 
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                         f'{int(bar.get_height())}', ha='center', va='bottom')
            
            # Display Graph
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.image(buf, caption="Session Analysis", width="stretch") # Using 'stretch' for compatibility
            plt.close(fig)
            
            # Reset Button
            if st.button("Start New Session"):
                st.session_state["webcam_results"] = None
                st.session_state["webrtc_key"] = f"sts-v1-{uuid.uuid4()}" # Generate new key
                del ctx 
                nuclear_cleanup() 
                st.rerun()

# ==========================================
# 📂 MODE 2: VIDEO FILE
# ==========================================
elif mode == "Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        MAX_FILE_SIZE = 100 * 1024 * 1024 # 100MB Limit
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File too large! Please upload a video smaller than 100MB. (Your file: {uploaded_file.size / (1024*1024):.1f} MB)")
        else:
            nuclear_cleanup() # Clean before processing
            
            # Hashing file to handle caching
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read(2 * 1024 * 1024) 
            file_hash = hashlib.md5(file_bytes).hexdigest()
            uploaded_file.seek(0) 

            session_id = st.session_state["user_session_id"]
            file_id = f"{session_id}_{file_hash}"
            
            output_filename = f"processed_{file_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            stats_key = f"stats_{file_id}"

            # --- Check Cache ---
            if os.path.exists(output_path) and stats_key in st.session_state:
                stats = st.session_state[stats_key]
                if st.session_state.get("just_processed") == file_id:
                    st.success("✅ Analysis Complete!")
                    del st.session_state["just_processed"]
                else:
                    st.success("✅ Analysis Complete! (Loaded from Cache)")

                # Display Results
                st.subheader("🎬 Analyzed Video")
                st.video(output_path)
                with open(output_path, "rb") as file:
                    st.download_button(label="⬇️ Download Analyzed Video", data=file, file_name="analyzed_sts.mp4", mime="video/mp4")

                st.divider()
                st.subheader("📊 Summary Report")
                total_reps = len(stats["reps"])
                correct_reps = sum(stats["reps"])
                accuracy = (correct_reps/total_reps*100) if total_reps > 0 else 0
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Reps", total_reps)
                col2.metric("Good Form", correct_reps)
                col3.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Graphs
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(stats["times"], stats["angles"], label='Knee Angle', color='blue')
                ax1.axhline(y=165, color='g', linestyle='--', label='Stand (165°)')
                ax1.axhline(y=100, color='r', linestyle='--', label='Sit (100°)')
                ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
                
                labels = ['Correct', 'Incorrect']; counts = [correct_reps, total_reps - correct_reps]
                bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
                ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
                for bar in bars: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
                
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.image(buf, caption="Session Analysis", width="stretch")
                plt.close(fig)

            # --- Process New Video ---
            elif os.path.exists(output_path) and stats_key not in st.session_state:
                 st.success("✅ Analysis Loaded from Cache!")
                 st.video(output_path)
                 with open(output_path, "rb") as file:
                    st.download_button(label="⬇️ Download Analyzed Video", data=file, file_name="analyzed_sts.mp4", mime="video/mp4")
                 st.info("ℹ️ Rename the file or re-upload to force new processing.")

            else:
                status_container = st.empty()
                log_container = st.empty()

                with status_container.container():
                     raw_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
                     raw_tfile.write(uploaded_file.read())
                     raw_tfile.close()

                cap = cv2.VideoCapture(raw_tfile.name)
                
                if not cap.isOpened():
                    st.error(f"Error: Could not open video file. Please ensure it is a standard MP4/MOV/AVI.")
                else:
                    logic = SitToStandLogic()
                    angle_data = []; time_data = []
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    
                    # Target dimensions
                    target_w = 1280 
                    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if original_w > target_w:
                        scale = target_w / original_w
                        target_h = int(original_h * scale)
                    else: target_w = original_w; target_h = original_h
                    
                    out = None
                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_bar = status_container.progress(0, text="Analyzing video frames... 0%")
                    
                    stop_flag = False
                    stop_reason = ""
                    last_log_time = time.time()

                    # Processing Loop
                    while cap.isOpened():
                        # Safety Cut
                        if frame_count % 30 == 0:
                            safe, mem_usage = check_memory_safe(1500)
                            if not safe:
                                stop_flag = True
                                stop_reason = f"{mem_usage:.1f} MB"
                                break

                        # Periodic Logging
                        current_time = time.time()
                        if current_time - last_log_time >= 3:
                            current_mem = get_current_memory_mb()
                            last_log_time = current_time
                        
                        ret, frame = cap.read()
                        if not ret: break

                        processed_img, angle, timestamp = logic.process_frame(frame)
                        
                        # Init Writer
                        if out is None:
                            h, w = processed_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(temp_output, fourcc, fps, (w, h))
                        
                        out.write(processed_img)
                        angle_data.append(angle)
                        time_data.append(timestamp)
                        frame_count += 1
                        
                        # Aggressive GC
                        if frame_count % 50 == 0: nuclear_cleanup()

                        # Update UI
                        if total_frames > 0:
                            progress = min(frame_count / total_frames, 1.0)
                            percent_text = f"Analyzing video frames... {int(progress * 100)}%"
                            progress_bar.progress(progress, text=percent_text)

                    # Cleanup
                    cap.release()
                    if out: out.release()
                    logic.close()
                    status_container.empty()
                    log_container.empty()
                    
                    if stop_flag:
                        st.error(f"⚠️ **Memory Limit Exceeded (1.5GB)!** System has been reset.")
                        st.error("🔄 **Please Refresh the Page to Continue.**")
                        
                        del cap, out, logic, angle_data, time_data
                        try:
                            if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)
                            if os.path.exists(temp_output): os.remove(temp_output)
                        except Exception: pass
                        
                        st.cache_resource.clear()
                        st.session_state.clear()
                        nuclear_cleanup()
                        st.stop()
                        
                    elif os.path.exists(temp_output) and os.path.getsize(temp_output) > 1000:
                        # Re-encode for browser compatibility
                        with st.spinner("💾 Finalizing video file..."):
                             os.system(f"ffmpeg -y -i {temp_output} -vcodec libx264 {output_path} -hide_banner -loglevel error")
                        
                        # Save Stats
                        st.session_state[stats_key] = {
                            "reps": logic.rep_quality_history,
                            "angles": angle_data,
                            "times": time_data
                        }
                        st.session_state["just_processed"] = file_id
                        st.rerun() 
                    
                    if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)
                    if os.path.exists(temp_output): os.remove(temp_output)
                    nuclear_cleanup()