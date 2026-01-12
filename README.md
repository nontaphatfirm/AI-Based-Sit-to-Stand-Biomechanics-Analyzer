# 🩺 AI-Based-Sit-to-Stand-Biomechanics-Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Stable%20v2.0.0-brightgreen)

**A computer vision application for automated clinical mobility assessment.**

This tool performs real-time kinematic analysis of **Sit-to-Stand (STS)** movements using markerless pose estimation. It utilizes **MediaPipe BlazePose** to evaluate posture quality, count repetitions, and generate biomechanical reports without the need for wearable sensors. 

**Now updated to v2.0.0 with 3D Biomechanics Logic & Auto-Side Detection!**

---

## 📥 Try It on WebApp
Web Version: Runs on iPad/iPhone/Android/PC (No installation required):
### **[🌐 Try Now (Streamlit Cloud)](https://ai-based-sts-biomechanics-analyzer.streamlit.app/)**

---

## 📥 Download Executable
Run the application on Windows locally without installing Python:
### **[⬇️ Download v2.0.0 (Windows .exe)](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer/releases)**

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

If you want to run the source code or modify it:

1. **Clone the repository**
   ```bash
   git clone [https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer.git](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer.git)
   cd AI-Based-Sit-to-Stand-Biomechanics-Analyzer

    ```

2. **Create a Virtual Environment (Recommended)**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    ```


3. **Install dependencies**
    ```bash
    pip install -r requirements.txt

    ```

*(Key libs: `opencv-python`, `mediapipe`, `numpy`, `matplotlib`, `psutil`)*
4. **Run the application**
    ```bash
    python main.py

    ```


---

## ⚙️ Technical Methodology

1. **Pose Estimation:** Extracts 33 3D-landmarks using **MediaPipe BlazePose** (CNN-based).
2. **Geometric Calculation:**
* **3D Knee Angle:** Calculated using vector dot product in 3D space to ensure perspective invariance.
* **Torso Lean:** Deviation of the Shoulder-Hip vector from the vertical axis.


3. **State Machine:**
* Operates as a finite state machine (`UP` ↔ `DOWN`).
* Repetitions are counted only upon full cycle completion.


4. **Error Correction (Reset Logic):**
* If a user commits a posture error but corrects it (returns to a perfect standing position), the system "forgives" the error for the current repetition to ensure fair assessment.



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
* **Project Status:** Active Development (v2.0.0)
* **GitHub:** [https://github.com/nontaphatfirm](https://github.com/nontaphatfirm)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### 📧 Contact

Project Link: [https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer)
