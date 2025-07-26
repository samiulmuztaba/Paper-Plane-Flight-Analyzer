# ✈️ Paper Plane Flight Analyzer

A Python + OpenCV-based tool to analyze and compare paper airplane flights using computer vision. Designed for physics research on how **wing design, folding complexity, launch angle, and center of mass** affect flight distance and stability.

---

## 🔧 Features

* **🎥 Video Input & Folder Scanning**

  * Load flight test videos from structured folders
  * Automatically detect and organize test runs (e.g., `CD/A15`, `CD/A15 M`)

* **📍 Flight Path Tracking**

  * Use OpenCV to detect and track airplane motion across frames
  * Record X-Y positions to generate full flight path

* **📊 Flight Metrics Calculation**

  * Distance traveled (horizontal)
  * Total airtime (duration in air)
  * Average speed
  * Stability scoring (based on trajectory straightness/smoothness)

* **📈 Trajectory Visualization**

  * Overlay trajectories from multiple runs
  * Highlight average path in bold
  * Transparent lines for individual flights

* **📁 Data Export**

  * Save detailed results (CSV): frame-by-frame positions, speed, etc.
  * Export annotated images and/or videos

* **📉 Graphing & Model Comparison**

  * Compare different models (e.g., CD, WG) and angles (0°, 15°, etc.)
  * Generate comparison graphs for:

    * Distance vs. Angle
    * Speed vs. Model
    * Stability vs. CoM

* **⚠️ Error Detection**

  * Warn if flight path seems abnormal
  * Warn for low visibility or bad camera angle

---

## 🗂 Folder Structure Example

```
flights/
│
├── CD/
│   ├── A0/
│   ├── A15/
│   └── A15 M/
├── WG/
│   ├── A15/
│   └── A30 M/
...
```

Each subfolder should contain 2–5 videos of that model and condition.

---

## 🧠 Use Cases

* Compare flight performance across different designs
* See how launch angle or added mass affects distance
* Investigate what makes a plane stable or unstable
* Help design new, optimized paper planes using real data

