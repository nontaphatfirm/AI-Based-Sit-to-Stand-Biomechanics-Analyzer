# 🩺 AI-Based-Sit-to-Stand-Biomechanics-Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Stable%20v2.0.0-brightgreen)

**A computer vision application for automated clinical mobility assessment.**

This tool performs real-time kinematic analysis of **Sit-to-Stand (STS)** movements using markerless pose estimation. It utilizes **MediaPipe BlazePose** to evaluate posture quality, count repetitions, and generate biomechanical reports without the need for wearable sensors. 

**Now updated to v2.0 with 3D Biomechanics Logic & Auto-Side Detection!**

---

## 📲 Try It on WebApp
Web Version: Runs on iPad/iPhone/Android/PC (No installation required):
### **[🌐 Try Now (Streamlit Cloud)](https://ai-based-sts-biomechanics-analyzer.streamlit.app/)**

---

## 📥 Download Executable
Run the application on Windows locally without installing Python:
### **[⬇️ Download v2.0 (Windows .exe)](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer/releases/tag/v2.0)**

---

## 🎬 Demo & Screenshots
(Image From v1.0)
| Real-time Analysis | Summary Dashboard |
|:---:|:---:|
| ![Analysis Demo](https://github.com/user-attachments/assets/d69cfd9a-d69f-4383-9138-ec054b69214c) | ![Graph Result](https://github.com/user-attachments/assets/3c1fb8c0-9447-4118-aa4e-e5333d1458e0) |
| *Live feedback on joint angles and posture* | *Post-session accuracy & kinematic graphs* |

---

## 🚀 What's New in v2.0

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

The system transforms raw video into clinical insights using a 4-step pipeline:

### 1. 3D Pose Estimation
We use **MediaPipe BlazePose** to track 33 body landmarks. Unlike basic 2D tools, we utilize **3D World Coordinates** $(x, y, z)$ for all angle calculations. This ensures accurate results even if the camera is placed at an oblique angle (e.g., 45° side view).

### 2. Auto-Side Detection
The system automatically determines the "Active Side" (Left vs. Right) by comparing:
* **Visibility Confidence:** Which leg is more clearly seen by the AI.
* **Projected Limb Size:** Which leg appears larger/closer to the camera.

### 3. Biomechanical Logic
* **Knee Angle:** Calculated using 3D Vector Geometry (Dot Product) to measure flexion/extension.
* **Torso Lean:** Measures the deviation of the trunk relative to the vertical gravity axis to detect unsafe leaning.

### 4. Smart Counting System
Uses a **Finite State Machine (FSM)** to track repetition cycles:
* **⬇️ Sit Phase:** Triggered when Knee Angle drops below **100°**.
* **⬆️ Stand Phase:** Triggered when Knee Angle exceeds **165°**.
* **🛡️ Debounce:** Filters out sensor noise to prevent double-counting.



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

Note: This repository is for educational purposes. It is currently maintained solely by the author.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### 📧 Contact

Project Link: [https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer)

