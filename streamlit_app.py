import os
import sys
import time
import gc
import uuid
import hashlib
import psutil
import ctypes # ☢️ Library สำหรับสั่งคืน RAM ระดับ OS

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
# ⚙️ Configuration
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

SMOOTH_WINDOW = 7
VISIBILITY_THRESHOLD = 0.5
LEAN_THRESHOLD = 45
MIN_FEET_RATIO = 0.5
MAX_FEET_RATIO = 1.4
BAD_POSTURE_DELAY = 3
INCOMPLETE_STAND_DELAY = 15

# ==========================================
# 🛡️ Memory Guard & Cleanup Tools
# ==========================================
def get_current_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def check_memory_safe(limit_mb=3000):
    current_mem_mb = get_current_memory_mb()
    if current_mem_mb > limit_mb:
        return False, current_mem_mb
    return True, current_mem_mb

def force_memory_cleanup():
    """☢️ The Nuclear Option: Force release memory back to OS"""
    gc.collect()
    try:
        # เฉพาะ Linux/Streamlit Cloud เท่านั้นที่ใช้คำสั่งนี้ได้
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass # ถ้าเป็น Windows ก็ข้ามไป

# ==========================================
# 📐 Helper Functions
# ==========================================
def calculate_angle(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def calculate_angle_3d(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b; bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_distance(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# ==========================================
# 🧠 Logic Class
# ==========================================
class SitToStandLogic:
    def __init__(self):
        # 🔄 REVERTED TO MODEL_COMPLEXITY=1 (Standard)
        # Reason: Streamlit Cloud blocks downloading the 'Lite' model (complexity=0) due to permissions.
        # We save RAM using frame skipping and resizing instead.
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)
        self.counter = 0; self.stage = None; self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = [] 
        self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
        self.current_side = "AUTO"

    def process_frame(self, image):
        if self.start_time is None: self.start_time = time.time()
        
        target_w = 640 # 📉 Optimize resolution
        h, w, c = image.shape
        if w > target_w:
            scale = target_w / w
            new_h = int(h * scale)
            image = cv2.resize(image, (target_w, new_h))
        else: target_w = w; new_h = h
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        
        current_angle = 0; feedback = "READY"; feedback_color = (0, 255, 0)
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks and results.pose_world_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                world_landmarks = results.pose_world_landmarks.landmark
                def get_2d(lm): return [lm.x, lm.y]
                def get_3d(lm): return [lm.x, lm.y, lm.z]
                
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                len_thigh_left = math.hypot(l_knee.x - l_hip.x, l_knee.y - l_hip.y)
                len_thigh_right = math.hypot(r_knee.x - r_hip.x, r_knee.y - r_hip.y)
                vis_left = (l_hip.visibility + l_knee.visibility) / 2
                vis_right = (r_hip.visibility + r_knee.visibility) / 2
                score_left = (len_thigh_left * 3.0) + vis_left
                score_right = (len_thigh_right * 3.0) + vis_right

                if score_left > score_right:
                    self.current_side = "LEFT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 23, 25, 27, 11
                else:
                    self.current_side = "RIGHT"; hip_idx, knee_idx, ankle_idx, shoulder_idx = 24, 26, 28, 12

                selected_knee_vis = landmarks[knee_idx].visibility
                if selected_knee_vis < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    hip_3d = get_3d(world_landmarks[hip_idx])
                    knee_3d = get_3d(world_landmarks[knee_idx])
                    ankle_3d = get_3d(world_landmarks[ankle_idx])

                    hip_2d = get_2d(landmarks[hip_idx])
                    knee_2d = get_2d(landmarks[knee_idx])
                    shoulder_2d = get_2d(landmarks[shoulder_idx])
                    
                    r_shoulder_2d = get_2d(landmarks[12])
                    l_shoulder_2d = get_2d(landmarks[11])
                    r_ankle_2d = get_2d(landmarks[28])
                    l_ankle_2d = get_2d(landmarks[27])

                    raw_angle = calculate_angle_3d(hip_3d, knee_3d, ankle_3d)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    angle_2d = calculate_angle(hip_2d, knee_2d, get_2d(landmarks[ankle_idx]))
                    torso_lean = calculate_vertical_angle(shoulder_2d, hip_2d)
                    shoulder_width = calculate_distance(l_shoulder_2d, r_shoulder_2d)
                    feet_width = calculate_distance(l_ankle_2d, r_ankle_2d)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    knee_px = tuple(np.multiply(knee_2d, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                    potential_bad_posture = False; temp_feedback = ""
                    if stance_ratio < MIN_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                    elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                    elif torso_lean > LEAN_THRESHOLD and (100 < angle_2d < 160): potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                    
                    if potential_bad_posture: self.bad_posture_counter += 1
                    else: self.bad_posture_counter = 0 
                    
                    potential_inc = False
                    if self.stage == 'up' and 140 < current_angle < 155: potential_inc = True
                    if potential_inc: self.incomplete_stand_counter += 1
                    else: self.incomplete_stand_counter = 0

                    if self.bad_posture_counter > BAD_POSTURE_DELAY:
                        feedback = temp_feedback; feedback_color = (0, 0, 255); self.current_rep_error = True 
                    elif self.incomplete_stand_counter > INCOMPLETE_STAND_DELAY:
                        feedback = "STAND UP FULLY!"; feedback_color = (0, 165, 255); self.current_rep_error = True
                    else: feedback = "GOOD FORM"; feedback_color = (0, 255, 0)

                    if current_angle > 165: 
                        self.stage = "up"
                        if feedback == "GOOD FORM": self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
                    
                    if current_angle < 100 and self.stage == 'up':
                        self.stage = "down"; self.counter += 1
                        if not self.current_rep_error: self.rep_quality_history.append(1) 
                        else: self.rep_quality_history.append(0) 
                        self.current_rep_error = False 

            except Exception: pass
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.rectangle(image, (0,0), (target_w, 85), (245,117,16), -1)
        x_rep = 20; x_feed = int(target_w * 0.25); x_acc = int(target_w * 0.65); x_time = int(target_w * 0.85)
        font_scale = 0.8 if target_w > 1000 else 0.6; font_thick = 2

        cv2.putText(image, 'REPS', (x_rep,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, str(self.counter), (x_rep-5,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*2, (255,255,255), font_thick)
        cv2.putText(image, 'FEEDBACK', (x_feed,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, feedback, (x_feed,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale, feedback_color, font_thick)
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        cv2.putText(image, 'ACC', (x_acc,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (x_acc,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.5, (255,255,255), font_thick)
        cv2.putText(image, 'TIME', (x_time,25), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (x_time,65), cv2.FONT_HERSHEY_SIMPLEX, font_scale*1.2, (255,255,255), font_thick)
        cv2.putText(image, f"Active: {self.current_side}", (20, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        return image, current_angle, current_time_seconds

# ==========================================
# 🌐 Streamlit Interface
# ==========================================
st.set_page_config(page_title="STS Analyzer", layout="wide")
st.title("🩺 AI-Based STS Biomechanics Analyzer")
st.markdown("**Web Version:** Runs on iPad/iPhone/Android/PC")

if "user_session_id" not in st.session_state:
    st.session_state["user_session_id"] = str(uuid.uuid4())[:8]

if "webcam_results" not in st.session_state: st.session_state["webcam_results"] = None

mode = st.radio("Select Input Source:", ("Webcam (Live)", "Video File"))

with st.expander("ℹ️ User Guide & Camera Setup (Click to open)", expanded=False):
    st.markdown(
        """
        ### 📸 Optimal Camera Positioning
        Our AI uses **3D Motion Analysis**, allowing it to track you from various angles. However, for the best results:
        
        1.  **📐 The "Sweet Spot" (45°):** Stand diagonally (approx. 45°) to the camera.
            * *Why?* This allows the AI to accurately measure **BOTH** your **Knee Angle** (for counting) and **Stance Width** (for posture check).
        2.  **📏 Full Body:** Ensure your **entire body** is visible from **Head to Toe** at all times.
        3.  **💡 Lighting:** Use a well-lit room and avoid wearing clothes that blend into the background.
        
        ---
        ### 📝 How to Use
        * **Webcam:** 1. Click **"Allow"** for camera permission.
            2. Click **"SELECT DEVICE"** to choose your camera (Front/Back).
            3. Click **"START"** to begin.
        * **Video File:** Upload a video (`.mp4`, max 200MB) recorded with the positioning above.
        * **Troubleshooting:** If the webcam freezes or fails to load, please switch to a **Mobile Hotspot**.
        """
    )

if mode == "Webcam (Live)":
    class VideoProcessor(VideoTransformerBase):
        def __init__(self): 
            self.logic = SitToStandLogic()
            self.angle_history = []
            self.time_history = []
        
        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)
                processed_img, angle, timestamp = self.logic.process_frame(img)
                self.angle_history.append(angle)
                self.time_history.append(timestamp)
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception: return frame
        
        def get_stats(self):
            return {
                "rep_quality_history": self.logic.rep_quality_history,
                "angle_history": self.angle_history,
                "time_history": self.time_history
            }

    st.info("💡 Instructions: Click 'START'. When finished, click 'STOP' to see results.")
    ctx = webrtc_streamer(
        key="sts-webcam-safe-v41", 
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 1280, "height": 720, "frameRate": 30}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        st.session_state["webcam_results"] = ctx.video_processor.get_stats()

    if not ctx.state.playing and st.session_state["webcam_results"]:
        data = st.session_state["webcam_results"]
        rep_history = data["rep_quality_history"]
        if len(rep_history) > 0 or len(data["angle_history"]) > 0:
            st.divider()
            st.subheader("📊 Session Summary")
            total_reps = len(rep_history)
            correct_reps = sum(rep_history)
            accuracy = (correct_reps/total_reps*100) if total_reps > 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Reps", total_reps)
            col2.metric("Good Form", correct_reps)
            col3.metric("Accuracy", f"{accuracy:.1f}%")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax1.plot(data["time_history"], data["angle_history"], label='Knee Angle', color='blue')
            
            ax1.axhline(y=165, color='g', linestyle='--', label='Stand (165°)')
            ax1.axhline(y=100, color='r', linestyle='--', label='Sit (100°)')
            
            ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
            
            labels = ['Correct', 'Incorrect']; counts = [correct_reps, total_reps - correct_reps]
            bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
            ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
            for bar in bars: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
            st.pyplot(fig)
            plt.close(fig)
            
            if st.button("Start New Session"):
                st.session_state["webcam_results"] = None
                gc.collect()
                st.rerun()

elif mode == "Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        MAX_FILE_SIZE = 200 * 1024 * 1024
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"❌ File too large! Please upload a video smaller than 200MB. (Your file: {uploaded_file.size / (1024*1024):.1f} MB)")
        else:
            # 🧹 PRE-CLEANUP
            force_memory_cleanup()
            
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read(2 * 1024 * 1024) 
            file_hash = hashlib.md5(file_bytes).hexdigest()
            uploaded_file.seek(0) 

            session_id = st.session_state["user_session_id"]
            file_id = f"{session_id}_{file_hash}"
            
            output_filename = f"processed_{file_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            stats_key = f"stats_{file_id}"

            if os.path.exists(output_path) and stats_key in st.session_state:
                stats = st.session_state[stats_key]
                if st.session_state.get("just_processed") == file_id:
                    st.success("✅ Analysis Complete!")
                    del st.session_state["just_processed"]
                else:
                    st.success("✅ Analysis Complete! (Loaded from Cache)")

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
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
                ax1.plot(stats["times"], stats["angles"], label='Knee Angle', color='blue')
                
                ax1.axhline(y=165, color='g', linestyle='--', label='Stand (165°)')
                ax1.axhline(y=100, color='r', linestyle='--', label='Sit (100°)')
                
                ax1.set_title('Knee Angle Movement Analysis'); ax1.grid(True); ax1.legend()
                labels = ['Correct', 'Incorrect']; counts = [correct_reps, total_reps - correct_reps]
                bars = ax2.bar(labels, counts, color=['#28a745', '#dc3545'])
                ax2.set_title('Repetition Quality'); ax2.set_ylabel('Count')
                for bar in bars: ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')
                st.pyplot(fig)
                plt.close(fig)

            elif os.path.exists(output_path) and stats_key not in st.session_state:
                 st.success("✅ Analysis Loaded from Cache!")
                 st.video(output_path)
                 with open(output_path, "rb") as file:
                    st.download_button(label="⬇️ Download Analyzed Video", data=file, file_name="analyzed_sts.mp4", mime="video/mp4")
                 st.info("ℹ️ Rename the file or re-upload to force new processing.")

            else:
                status_container = st.empty()
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

                    while cap.isOpened():
                        # 🛡️ MEMORY GUARD CHECK: 3000MB Limit
                        if frame_count % 30 == 0:
                            safe, mem_usage = check_memory_safe(limit_mb=3000)
                            if not safe:
                                stop_flag = True
                                stop_reason = f"{mem_usage:.1f} MB"
                                break
                        
                        ret, frame = cap.read()
                        if not ret: break

                        # ⏭️ FRAME SKIPPING: Process every 3rd frame
                        if frame_count % 3 != 0:
                            frame_count += 1
                            continue

                        processed_img, angle, timestamp = logic.process_frame(frame)
                        if out is None:
                            h, w = processed_img.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(temp_output, fourcc, fps / 3, (w, h))
                        out.write(processed_img)
                        angle_data.append(angle)
                        time_data.append(timestamp)
                        frame_count += 1
                        
                        # Aggressive GC
                        if frame_count % 50 == 0: force_memory_cleanup()

                        if total_frames > 0:
                            progress = min(frame_count / total_frames, 1.0)
                            percent_text = f"Analyzing video frames... {int(progress * 100)}%"
                            progress_bar.progress(progress, text=percent_text)

                    cap.release()
                    if out: out.release()
                    status_container.empty()
                    
                    if stop_flag:
                        st.error(f"⚠️ **System Warning:** Stopped due to memory limit! (Current: {stop_reason})")
                        st.warning("ℹ️ **Action:** RAM has been aggressively cleared.")
                        
                        # 🚨 FORCE RELEASE RESOURCES & CLEAN MEMORY
                        del cap, out, logic, angle_data, time_data
                        try:
                            if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)
                            if os.path.exists(temp_output): os.remove(temp_output)
                        except Exception: pass
                        
                        force_memory_cleanup() # ☢️ Nuclear Clean
                        
                    elif os.path.exists(temp_output) and os.path.getsize(temp_output) > 1000:
                        with st.spinner("💾 Finalizing video file..."):
                             os.system(f"ffmpeg -y -i {temp_output} -vcodec libx264 {output_path} -hide_banner -loglevel error")
                        st.session_state[stats_key] = {
                            "reps": logic.rep_quality_history,
                            "angles": angle_data,
                            "times": time_data
                        }
                        st.session_state["just_processed"] = file_id
                        st.rerun() 
                    
                    # Normal Cleanup
                    if os.path.exists(raw_tfile.name): os.remove(raw_tfile.name)
                    if os.path.exists(temp_output): os.remove(temp_output)
                    force_memory_cleanup()