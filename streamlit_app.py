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
# 📐 Helper Functions
# ==========================================
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

# ==========================================
# 🧠 Logic Class (Foreshortening Detection)
# ==========================================
class SitToStandLogic:
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.counter = 0; self.stage = None; self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = [] 
        self.current_rep_error = False; self.bad_posture_counter = 0; self.incomplete_stand_counter = 0
        self.current_side = "DETECTING..." 

    def process_frame(self, image):
        if self.start_time is None: self.start_time = time.time()
        
        target_w = 640
        h, w, c = image.shape
        scale = target_w / w
        new_h = int(h * scale)
        image = cv2.resize(image, (target_w, new_h))
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image.flags.writeable = True
        
        current_angle = 0; feedback = "READY"; feedback_color = (0, 255, 0)
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # =========================================================
                # 🧠 LOGIC: Longest Thigh Selection (แก้ Foreshortening)
                # =========================================================
                def get_raw(lm): return [lm.x, lm.y]
                
                # 1. Get Landmarks
                l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]

                # 2. Calculate "Apparent Thigh Length" (ระยะ 2D บนหน้าจอ)
                # สูตร: ระยะห่างจริงระหว่างจุด Hip กับ Knee (รวมแกน X และ Y)
                # ถ้าขาไหนหันหน้าหากล้อง ค่านี้จะต่ำมาก (เพราะจุดมันซ้อนกัน)
                len_thigh_left = math.hypot(l_knee.x - l_hip.x, l_knee.y - l_hip.y)
                len_thigh_right = math.hypot(r_knee.x - r_hip.x, r_knee.y - r_hip.y)

                # 3. Get Visibility
                vis_left = (l_hip.visibility + l_knee.visibility) / 2
                vis_right = (r_hip.visibility + r_knee.visibility) / 2

                # 4. Final Score (ให้น้ำหนักความยาวขา 80%, ความชัด 20%)
                # ขาที่ยาวย่อมดีกว่าขาที่สั้นเสมอ ในมุมมอง 2D
                score_left = (len_thigh_left * 3.0) + vis_left
                score_right = (len_thigh_right * 3.0) + vis_right

                # 5. Select Best Side
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

                # Check minimum visibility
                selected_knee_vis = landmarks[knee_idx].visibility
                
                if selected_knee_vis < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    hip_raw = get_raw(landmarks[hip_idx])
                    knee_raw = get_raw(landmarks[knee_idx])
                    ankle_raw = get_raw(landmarks[ankle_idx])
                    shoulder_raw = get_raw(landmarks[shoulder_idx])
                    
                    r_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                    l_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                    r_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                    l_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                    raw_angle = calculate_angle(hip_raw, knee_raw, ankle_raw)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    torso_lean = calculate_vertical_angle(shoulder_raw, hip_raw)
                    shoulder_width = calculate_distance(l_shoulder_raw, r_shoulder_raw)
                    feet_width = calculate_distance(l_ankle_raw, r_ankle_raw)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    knee_px = tuple(np.multiply(knee_raw, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Debug Info: Show which leg is active
                    cv2.putText(image, f"Active: {self.current_side}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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

        # --- UI Overlay ---
        cv2.rectangle(image, (0, new_h - 60), (target_w, new_h), (0,0,0), -1)
        cv2.putText(image, f"REPS: {self.counter}", (20, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(image, feedback, (180, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2)
        
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        cv2.putText(image, f"ACC: {int(current_acc)}%", (target_w - 140, new_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        return image, current_angle, current_time_seconds

# ==========================================
# 🌐 Streamlit Interface (Main App)
# ==========================================
st.set_page_config(page_title="STS Analyzer", layout="wide")
st.title("🩺 AI-Based STS Biomechanics Analyzer")
st.markdown("**Web Version:** Runs on iPad/iPhone/Android/PC")

mode = st.radio("Select Input Source:", ("Webcam (Live)", "Video File"))

if mode == "Webcam (Live)":
    class VideoProcessor(VideoTransformerBase):
        def __init__(self): self.logic = SitToStandLogic()
        def recv(self, frame):
            try:
                time.sleep(0.01)
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1) # Mirror
                processed_img, _, _ = self.logic.process_frame(img)
                return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
            except Exception as e: return frame

    st.info("💡 Instructions: Click 'START'. If WiFi fails, try using Mobile Hotspot.")
    
    ctx = webrtc_streamer(
        key="sts-webcam-final-v14", # New Key
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"width": 320, "height": 240, "frameRate": 15}, "audio": False},
        async_processing=True,
    )

elif mode == "Video File":
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            logic = SitToStandLogic()
            angle_data = []; time_data = []
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            target_w = 640
            original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            scale = target_w / original_w
            target_h = int(original_h * scale)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(temp_output.name, fourcc, fps, (target_w, target_h))

            progress_bar = st.progress(0)
            status_text = st.empty()
            st.info("⏳ Processing video... Please wait.")
            
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

            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()
            
            converted_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            os.system(f"ffmpeg -y -i {temp_output.name} -vcodec libx264 {converted_output.name}")

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