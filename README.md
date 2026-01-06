# 🩺 AI-Based-Sit-to-Stand-Biomechanics-Analyzer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

**A computer vision application for automated clinical mobility assessment.**

This tool performs real-time kinematic analysis of **Sit-to-Stand (STS)** movements using markerless pose estimation. It utilizes **MediaPipe BlazePose** to evaluate posture quality, count repetitions, and generate biomechanical reports without the need for wearable sensors. Designed for physical therapy, geriatric care, and fitness assessment.

---

## 📥 Download Executable
Run the application on Windows without installing Python:
### **[⬇️ Download v1.0 (Windows .exe)](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer/releases)**

---

## 🎬 Demo & Screenshots

| Real-time Analysis | Summary Report |
|:---:|:---:|
| ![Analysis Demo](https://github.com/user-attachments/assets/d69cfd9a-d69f-4383-9138-ec054b69214c) | ![Graph Result](https://github.com/user-attachments/assets/3c1fb8c0-9447-4118-aa4e-e5333d1458e0) |
| *Live feedback on joint angles and posture* | *Post-session accuracy & kinematic graphs* |

---

## 🚀 Key Features

### 🧠 Intelligent Motion Analysis
* **Markerless Tracking:** Precision body landmark detection using **MediaPipe**.
* **Smart Resize:** Automatically scales the video feed to fit 85% of your screen size (supports 4K/Vertical videos).
* **Universal Counting Logic:** Automatically detects "Sit" (< 85°) and "Stand" (> 160°) phases (unlimited repetitions).

### 🛡️ Posture Analysis (Quality Control)
The system uses rule-based logic to detect common biomechanical errors:
* **⚠️ Torso Lean:** Detects excessive forward leaning (trunk flexion > 45°).
* **⚠️ Stance Width:** Monitors foot placement ratio (Narrow/Wide stance).
* **⚠️ Incomplete Extension:** Detects failure to stand fully upright (Knee angle 140°-155°).
* **Debounce Logic:** Prevents false positives by requiring error persistence (0.1s - 0.5s).

### 📊 Advanced Reporting
* **Accuracy Calculation:** Calculates the percentage of "Good Form" repetitions.
* **Data Visualization:** Generates dual-plot graphs (Kinematics + Pass/Fail Bar Chart).
* **CSV Export:** Auto-saves raw time-series data for further research.
* **Video Export:** Saves the analyzed video with overlay for review.

### 💻 Smart UI/UX
* **Dual Input Support:** Works with **Webcam** (Live) or **Video Files**.
* **Dynamic Overlay:** UI elements (Timer, Reps, Accuracy) auto-adjust positions based on window size.
* **Auto-Organized Output:** Creates timestamped folders for every session.

---

## 🛠️ Installation (For Developers)

If you want to run the source code or modify it:

1. **Clone the repository**
   ```bash
   git clone https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer.git
   cd AI-Based-Sit-to-Stand-Biomechanics-Analyzer
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Requires: `opencv-python`, `mediapipe`, `numpy`, `matplotlib`, `tkinter`)*

3. **Run the application**
    ```bash
    python main.py
    ```



---

## ⚙️ Technical Methodology

1. **Pose Estimation:** Extracts 33 3D-landmarks using a CNN-based model (BlazePose).
2. **Geometric Calculation:**
    * **Knee Angle:** Calculated using vector dot product of  and  (A=Hip, B=Knee, C=Ankle).
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
SitToStand_Webcam_2024-01-07_14-30-00/
├── processed_video.mp4      # Analyzed video with overlay
├── motion_data.csv          # Raw data (Timestamp, Knee Angle)
└── summary_report.png       # Final visual graph

```

---

## 👨‍💻 Author

**Nontapat Auetrongjit**
* **Role:** Lead Developer & Researcher
* **Project Status:** Personal Project / Academic Research
* **GitHub:** [https://github.com/nontaphatfirm](https://github.com/nontaphatfirm)

*Note: This repository is for educational purposes. It is currently maintained solely by the author.*

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

---

### 📧 Contact

Project Link: [https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer](https://github.com/nontaphatfirm/AI-Based-Sit-to-Stand-Biomechanics-Analyzer)
