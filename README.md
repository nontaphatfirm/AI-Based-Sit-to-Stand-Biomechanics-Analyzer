# 🩺 AI-Based-Sit-to-Stand-Biomechanics-Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Stable%20v2.0.0-brightgreen)

**A computer vision application for automated clinical mobility assessment.**

This tool performs real-time kinematic analysis of **Sit-to-Stand (STS)** movements using markerless pose estimation. It utilizes **MediaPipe BlazePose** to evaluate posture quality, count repetitions, and generate biomechanical reports without the need for wearable sensors. 

**Now updated to v2.0.0 with 3D Biomechanics Logic & Auto-Side Detection!**

---

## 📲 Try It on WebApp
Web Version: Runs on iPad/iPhone/Android/PC (No installation required):
### **[🌐 Try Now (Streamlit Cloud)](https://ai-based-sts-biomechanics-analyzer.streamlit.app/)**

---

## 📥 Download Executable
Run the application on Windows locally without installing Python:
### **[⬇️ Download v2.0 (Windows .exe)](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer/releases)**

---

## 🎬 Demo & Screenshots

| Real-time Analysis | Summary Dashboard |
|:---:|:---:|
| ![Analysis Demo](https://github.com/user-attachments/assets/d69cfd9a-d69f-4383-9138-ec054b69214c) | ![Graph Result](https://github.com/user-attachments/assets/3c1fb8c0-9447-4118-aa4e-e5333d1458e0) |
| *Live feedback on joint angles and posture* | *Post-session accuracy & kinematic graphs* |

---

## 🚀 What's New in v2.0.0

### 🦵 Auto-Side Detection (Dual-Leg Support)
* The system now analyzes **both left and right legs** simultaneously.
* Automatically selects the "Active Side" based on visibility and movement quality. No manual setup required.

### 🧠 3D Biomechanics Logic
* Upgraded from 2D screen coordinates to **3D world coordinates**.
* Provides accurate knee angle calculations even when the camera is positioned at a 45-degree angle (the optimal viewing spot).

### 🛡️ Smart Resource Management
* **Auto-Nuke RAM:** Intelligent memory management system that prevents crashes during long sessions by automatically cleaning up resources.

---

## ✨ Key Features

### 🔍 Intelligent Motion Analysis
* **Markerless Tracking:** Precision body landmark detection using **MediaPipe**.
* **Smart Resize:** Automatically scales the video feed to fit 85% of your screen size (supports 4K/Vertical videos).
* **Universal Counting Logic:** Automatically detects "Sit" (< 100°) and "Stand" (> 165°) phases.

### ⚠️ Posture Analysis (Quality Control)
The system uses rule-based logic to detect common biomechanical errors:
* **Torso Lean:** Detects excessive forward leaning (trunk flexion > 45°).
* **Stance Width:** Monitors foot placement ratio (Narrow/Wide stance).
* **Incomplete Extension:** Detects failure to stand fully upright.
* **Debounce Logic:** Prevents false positives by requiring error persistence.

### 📊 Advanced Reporting
* **Session Dashboard:** Auto-generates a comprehensive summary with Kinematics Graph and Accuracy Bar Chart.
* **CSV Export:** Auto-saves raw time-series data for further research.
* **Video Export:** Saves the analyzed video with overlay for review.

### 💻 Smart UI/UX
* **Dual Input Support:** Works with **Webcam** (Live) or **Video Files**.
* **On-Screen Instructions:** Clear "Press 'q' to Finish" prompts.
* **Auto-Organized Output:** Creates timestamped folders for every session.

---

## 🛠️ Installation (For Developers)

**⚠️ Important Requirement:** This project is strictly compatible with **Python 3.10** and **MediaPipe 0.10.9**. (Newer Python versions like 3.11/3.12 are NOT supported by this MediaPipe version).


1. **Clone the repository**
   ```bash
   git clone https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer.git

   cd AI-Based-Sit-to-Stand-Biomechanics-Analyzer

    ```


2. **Setup Python 3.10 Environment**
Make sure you have Python 3.10 installed on your system.
    ```bash
    # Create Virtual Environment with Python 3.10
    py -3.10 -m venv venv

    # Activate Environment
    .\venv\Scripts\activate

    ```


3. **Install dependencies**
    ```bash
    pip install -r requirements-dev.txt

    ```


4. **Run the application**
    ```bash
    python main.py

    ```


---

## ⚙️ Technical Methodology

The system operates on a sophisticated pipeline designed to transform raw video frames into actionable clinical insights.

### 1. Pose Estimation & Signal Processing
* **Model Architecture:** Utilizes **MediaPipe BlazePose** (GHUM 3D), a high-fidelity topology model that predicts 33 body landmarks in real-time.
* **Coordinate Systems:**
    * **2D Normalized Coordinates $(x, y)$:** Used for UI overlay and posture metrics relative to the camera frame (e.g., Stance Width).
    * **3D World Coordinates $(x, y, z)$:** Used for angular kinematics to ensure **perspective invariance**, allowing accurate analysis even from oblique camera angles (45°).
* **Signal Smoothing:** A **Moving Average Filter** (Window Size $N=7$) is applied to the angular data stream to eliminate jitter and sensor noise.

### 2. Algorithmic Side Selection (Auto-Side Detection)
To support unconstrained usage, the system automatically determines the "Active Side" (Left vs. Right) using a heuristic scoring algorithm:
$$Score_{side} = (L_{thigh} \times 3.0) + V_{avg}$$
Where:
* $L_{thigh}$: Euclidean distance between Hip and Knee in 2D space (Projected length).
* $V_{avg}$: Average visibility score of Hip and Knee landmarks provided by the model confidence.
* **Logic:** The side with the higher score is selected as the active tracking side for the session.

### 3. Biomechanical Calculations
The core kinematic analysis relies on vector geometry:

* **Knee Flexion Angle ($\theta_{knee}$):** Calculated using the **3D Vector Dot Product** formula to measure the angle between the Thigh vector ($\vec{BA}$) and Shank vector ($\vec{BC}$):
    $$\theta_{knee} = \arccos \left( \frac{\vec{BA} \cdot \vec{BC}}{||\vec{BA}|| \cdot ||\vec{BC}||} \right)$$
    *(Where $B$=Knee, $A$=Hip, $C$=Ankle in 3D space)*

* **Trunk Flexion (Torso Lean):** Calculated as the deviation of the trunk vector relative to the vertical gravity axis ($Y_{axis}$).

### 4. Finite State Machine (FSM)
Repetition counting is managed by a robust state machine to prevent false counts:

1.  **State: STANDING** (Initial)
    * Transition Condition: Knee Angle $> 165^\circ$
2.  **State: SITTING** (Trigger Count)
    * Transition Condition: Knee Angle $< 100^\circ$ AND Previous State was `STANDING`.
    * **Action:** Increment Counter ($+1$).
3.  **Error Handling (Debounce Logic):**
    * Posture errors (e.g., Lean, Asymmetry) must persist for $> 3$ consecutive frames to trigger a warning, preventing flickering feedback.



---

## 📂 Output Structure

The program automatically creates a new folder for each session:

```text
SitToStand_Webcam_2026-01-12_14-30-00/
├── processed_video.mp4      # Analyzed video with overlay
├── motion_data.csv          # Raw data (Timestamp, Knee Angle)
└── summary_report.png       # Final visual graph dashboard

```

---

## 👨‍💻 Author

**Nontapat Auetrongjit**

* **Role:** Lead Developer & Researcher
* **Project Status:** Active Development (v2.0)
* **GitHub:** [https://github.com/nontaphatfirm](https://github.com/nontaphatfirm)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### 📧 Contact

Project Link: [https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer)

