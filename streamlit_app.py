import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
import numpy as np
import av
import math
import time
from collections import deque
import tempfile

# ==========================================
# ⚙️ Configuration & Thresholds
# ==========================================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Filtering Constants
SMOOTH_WINDOW = 7      # Frame window for moving average
VISIBILITY_THRESHOLD = 0.6 

# Safety & Posture Thresholds
LEAN_THRESHOLD = 45    # Max allowed trunk flexion (degrees)
MIN_FEET_RATIO = 0.5   # Min stance width ratio
MAX_FEET_RATIO = 1.4   # Max stance width ratio

# Debounce Delays (Frame Counts)
BAD_POSTURE_DELAY = 3   # Approx 0.1s for red alert
INCOMPLETE_STAND_DELAY = 15 # Approx 0.5s for orange alert

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
    """ Calculate Euclidean distance between two points """
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def calculate_vertical_angle(a, b):
    """ Calculate angle relative to the vertical axis (for torso lean) """
    return np.degrees(np.arctan2(abs(a[0] - b[0]), abs(a[1] - b[1])))

# ==========================================
# 🧠 Logic Class (Encapsulates core logic for Web)
# ==========================================
class SitToStandLogic:
    def __init__(self):
        # Initialize Pose Model
        self.pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        
        # System Variables
        self.counter = 0 
        self.stage = None 
        self.start_time = None
        self.angle_buffer = deque(maxlen=SMOOTH_WINDOW)
        self.rep_quality_history = [] 
        
        # Logic Flags
        self.current_rep_error = False 
        self.bad_posture_counter = 0        
        self.incomplete_stand_counter = 0 

    def process_frame(self, image):
        # Start timer on first frame
        if self.start_time is None:
            self.start_time = time.time()
            
        # Resize for web performance (Standard mobile width)
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
        
        # Logic Variables
        current_angle = 0 
        feedback = "READY"
        feedback_color = (0, 255, 0) # Green
        current_time_seconds = time.time() - self.start_time

        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Check visibility
                vis_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility
                vis_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility
                
                if vis_hip < VISIBILITY_THRESHOLD or vis_knee < VISIBILITY_THRESHOLD:
                    feedback = "LOW VISIBILITY"; feedback_color = (0, 0, 255)
                else:
                    # Helper to get raw coordinates (0-1)
                    def get_raw(lm): return [lm.x, lm.y]

                    # Extract Keypoints (Raw for calculation)
                    r_hip_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
                    r_knee_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
                    r_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
                    r_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
                    l_shoulder_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
                    l_ankle_raw = get_raw(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

                    # Calculate Biomechanics
                    raw_angle = calculate_angle(r_hip_raw, r_knee_raw, r_ankle_raw)
                    self.angle_buffer.append(raw_angle)
                    current_angle = sum(self.angle_buffer) / len(self.angle_buffer)
                    
                    torso_lean = calculate_vertical_angle(r_shoulder_raw, r_hip_raw)
                    shoulder_width = calculate_distance(l_shoulder_raw, r_shoulder_raw)
                    feet_width = calculate_distance(l_ankle_raw, r_ankle_raw)
                    stance_ratio = 0 if shoulder_width == 0 else feet_width / shoulder_width

                    # Visualize Angle on Knee
                    r_knee_px = tuple(np.multiply(r_knee_raw, [target_w, new_h]).astype(int))
                    cv2.putText(image, str(int(current_angle)), r_knee_px, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                    # ==================================
                    # 🛡️ SAFETY CHECK LOGIC
                    # ==================================
                    potential_bad_posture = False; temp_feedback = ""
                    
                    if stance_ratio < MIN_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "NARROW STANCE!"
                    elif stance_ratio > MAX_FEET_RATIO and current_angle > 150: potential_bad_posture = True; temp_feedback = "WIDE STANCE!"
                    elif torso_lean > LEAN_THRESHOLD and (100 < current_angle < 160): potential_bad_posture = True; temp_feedback = "DONT LEAN!"
                    
                    if potential_bad_posture: self.bad_posture_counter += 1
                    else: self.bad_posture_counter = 0 
                    
                    potential_incomplete_stand = False
                    if self.stage == 'up' and 140 < current_angle < 155: potential_incomplete_stand = True
                    
                    if potential_incomplete_stand: self.incomplete_stand_counter += 1
                    else: self.incomplete_stand_counter = 0

                    if self.bad_posture_counter > BAD_POSTURE_DELAY:
                        feedback = temp_feedback; feedback_color = (0, 0, 255); self.current_rep_error = True 
                    elif self.incomplete_stand_counter > INCOMPLETE_STAND_DELAY:
                        feedback = "STAND UP FULLY!"; feedback_color = (0, 165, 255); self.current_rep_error = True
                    else:
                        feedback = "GOOD FORM"; feedback_color = (0, 255, 0)

                    # ==================================
                    # 🔢 REPETITION COUNTING LOGIC
                    # ==================================
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

            # Draw Skeleton
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # ==================================
        # 🎨 UI Display (Dynamic Positioning)
        # ==================================
        cv2.rectangle(image, (0,0), (target_w, 85), (245,117,16), -1)
        
        # Calculate positions
        x_rep = 15
        x_feed = int(target_w * 0.2)
        x_acc = int(target_w * 0.65)
        x_time = int(target_w * 0.85)

        # 1. REPS
        cv2.putText(image, 'REPS', (x_rep,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, str(self.counter), (x_rep-5,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        
        # 2. FEEDBACK
        cv2.putText(image, 'FEEDBACK', (x_feed,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, feedback, (x_feed,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2)
        
        # 3. ACCURACY
        current_acc = 0.0
        if len(self.rep_quality_history) > 0: current_acc = (sum(self.rep_quality_history) / len(self.rep_quality_history)) * 100
        cv2.putText(image, 'ACC', (x_acc,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{int(current_acc)}%", (x_acc,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        # 4. TIME
        cv2.putText(image, 'TIME', (x_time,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        cv2.putText(image, f"{current_time_seconds:.1f}s", (x_time,65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
        return image

# ==========================================
# 🌐 Streamlit Interface (Web UI)
# ==========================================
st.set_page_config(page_title="STS Analyzer", layout="wide")
st.title("🩺 AI-Based STS Biomechanics Analyzer")
st.markdown("**Web Version:** Runs on iPad/iPhone/Android/PC")

# Select Input Mode
mode = st.radio("Select Input Source:", ("Webcam (Live)", "Video File"))

if mode == "Webcam (Live)":
    # ----------------------------------------
    # WEBCAM MODE (Using streamlit-webrtc)
    # ----------------------------------------
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.logic = SitToStandLogic()
        
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            # Flip if it's webcam (Mirror effect)
            img = cv2.flip(img, 1)
            
            # Process Frame
            processed_img = self.logic.process_frame(img)
            
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

    ctx = webrtc_streamer(
        key="sts-analyzer",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

    st.info("💡 Instructions: Click 'START' and allow camera access. Stand sideways to the camera so your full body is visible.")

elif mode == "Video File":
    # ----------------------------------------
    # VIDEO FILE MODE
    # ----------------------------------------
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty() # Placeholder for video frame
        
        logic = SitToStandLogic()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Process Frame (No Flip for Video Files)
            processed_img = logic.process_frame(frame)
            
            # Display in Streamlit (Convert BGR to RGB)
            st_frame.image(processed_img, channels="BGR", use_column_width=True)
            
        cap.release()