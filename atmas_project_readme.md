# ATMAS: Autonomous Traffic Monitoring and Analysis System

[![Patent](https://img.shields.io/badge/Patent-Indian%20Patent%20No.%20549991-gold.svg)](https://ipindia.gov.in/)
[![Application](https://img.shields.io/badge/Application%20No.-202221057506-blue.svg)](https://ipindia.gov.in/)
[![Hardware](https://img.shields.io/badge/Target%20Edge-NVIDIA%20Jetson%20Nano-76B900.svg)](https://developer.nvidia.com/embedded/jetson-nano)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-lightgrey.svg)](#license)

> **Autonomous Traffic Monitoring and Analysis System (ATMAS)** is an edge-native, real-time computer vision and kinematic profiling pipeline developed under the mentorship of **Prof. Udit Jain** at the **Visvesvaraya National Institute of Technology (VNIT), Nagpur**. 
>
> The system transforms roadside monocular video streams into high-precision vehicle kinematic metrics (trajectories, metric velocities, and instantaneous acceleration profiles) using lightweight deep learning detectors and optical flow tracking deployed on resource-constrained embedded hardware (**NVIDIA Jetson Nano**). The extracted empirical telemetry is coupled with an autonomous highway engineering advisory engine built on **Indian Roads Congress (IRC)** standards to recommend roadway geometry revisions, lane capacity sizing, and traffic signage postings.

---

## Table of Contents
1. [Core Features](#core-features)
2. [System Architecture](#system-architecture)
3. [Mathematical & Theoretical Formulation](#mathematical--theoretical-formulation)
   - [Planar Homography Calibration](#1-planar-homography-calibration)
   - [Kinematics & Rolling Temporal Filtering](#2-kinematics--rolling-temporal-filtering)
   - [IRC Highway Engineering Advisory Engine](#3-irc-highway-engineering-advisory-engine)
4. [Hardware & Edge Deployment](#hardware--edge-deployment)
5. [Directory Layout](#directory-layout)
6. [Getting Started](#getting-started)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
   - [Quickstart with Google Colab](#quickstart-with-google-colab)
7. [Telemetry Schema](#telemetry-schema)
8. [Patent Information](#patent-information)
9. [Citation & Contact](#citation--contact)

---

## Core Features

- **Real-Time Edge Inference:** Engineered for embedded hardware (NVIDIA Jetson Nano, 4GB RAM, 128 Maxwell CUDA cores) using hardware-accelerated GStreamer pipelines and TensorRT FP16 quantization.
- **Monocular Ground Metric Calibration:** Implements planar homography ($H$) mapping pixel contact points to real-world Cartesian metric coordinates $(X, Y)$ without depth sensors or LiDAR.
- **Kinematic & Safety Anomaly Profiling:**
  - Continuous velocity ($km/h$) and instantaneous acceleration ($m/s^2$) estimation.
  - Automated rash driving identification: Over-speeding, abrupt braking ($a < -3.5\text{ m/s}^2$), and erratic swerving.
- **Automated Civil Highway Engineering Engine:**
  - **Empirical Operating Speeds:** Autonomous extraction of $V_{50}$, $V_{85}$, and $V_{98}$.
  - **Sight Distance Design (IRC:73 / IRC:86):** Real-time calculation of safe Stopping Sight Distance (SSD) and Intermediate Sight Distance (ISD).
  - **Capacity & Sizing (IRC:106-1990):** Stream conversion to Equivalent Passenger Car Units (PCU/hr), Volume-to-Capacity ($V/C$) evaluation, and automated lane-addition triggers.
  - **Corrective Signage Specifications (IRC:67-2012):** Autonomous placement advice for regulatory speed signs, cautionary hazard warnings, and kerbside friction controls.

---

## System Architecture

```
                      +-----------------------------+
                      |   IMX CSI Camera / Video   |
                      +--------------+--------------+
                                     |
                                     v [NVMM GStreamer Pipeline]
                      +-----------------------------+
                      | Frame Ingestion & Resizing  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |   YOLO Detector (TensorRT)  |
                      |  (Cars, Bikes, Trucks, etc) |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
       +----------------------------+  +----------------------------+
       |   Optical Flow / Centroid  |  |    Planar Homography (H)   |
       |      Persistent Tracking   |  |   Pixel -> Ground Plane (m)|
       +-------------+--------------+  +-------------+--------------+
                     |                               |
                     +---------------+---------------+
                                     |
                                     v
                      +-----------------------------+
                      | Rolling-Window Kinematics   |
                      | (Velocity & Acceleration)   |
                      +--------------+--------------+
                                     |
                     +---------------+---------------+
                     |                               |
                     v                               v
       +----------------------------+  +----------------------------+
       |  Safety Rule Alert Engine  |  |   IRC Civil Road Advisory  |
       |  - Overspeeding Violations |  |   - V85 & Safe Sight Dist. |
       |  - Harsh Braking / Jerk    |  |   - PCU Flow & V/C Ratio   |
       |  - Congestion / LOS Index  |  |   - Lane & Sign Sizing     |
       +----------------------------+  +----------------------------+
```

---

## Mathematical & Theoretical Formulation

### 1. Planar Homography Calibration
Monocular video streams project a 3D world onto a 2D image plane, inducing nonlinear perspective distortion. To convert pixel displacement into physical distance, four surveyed roadway reference marks with known physical dimensions are used to compute a $3 \times 3$ homography transformation matrix $H$:

$$\begin{bmatrix} X_{world} \\ Y_{world} \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u_{pixel} \\ v_{pixel} \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u_{pixel} \\ v_{pixel} \\ 1 \end{bmatrix}$$

The bottom-center point of each vehicle bounding box $(\frac{x_1 + x_2}{2}, y_2)$—representing the tire-pavement interface—is projected through $H$ to obtain ground coordinates $(X, Y)$ in meters.

### 2. Kinematics & Rolling Temporal Filtering
To suppress sensor jitter and bounding-box flicker, kinematics are calculated over a rolling temporal window:

- **Velocity ($v$):**
  $$v(t) = \frac{\sqrt{\Delta X^2 + \Delta Y^2}}{\Delta t} \times 3.6 \quad [\text{km/h}]$$
- **Instantaneous Acceleration ($a$):**
  $$a(t) = \frac{v_{\text{current}} - v_{\text{prior}}}{\Delta t} \quad [\text{m/s}^2]$$
- **Hazard Triggers:** An anomaly is flagged if $v > v_{\text{limit}}$ or $|a| > 3.5\text{ m/s}^2$ (standard threshold for harsh braking / aggressive throttle inputs).

### 3. IRC Highway Engineering Advisory Engine
- **Stopping Sight Distance (SSD - IRC:73 / IRC:86):**
  $$\text{SSD} = 0.278 \cdot V_{85} \cdot t + \frac{V_{85}^2}{254 \cdot (f \pm 0.01n)}$$
  *(Where driver reaction time $t = 2.5\text{ s}$, friction factor $f \approx 0.35\text{--}0.40$, and road grade $n$).*
- **Volume & Capacity (IRC:106-1990):**
  $$\text{Hourly Volume } (V) = \frac{\sum (\text{Vehicle Class}_i \times \text{PCU}_i)}{\Delta t_{\text{hours}}}$$
  $$\text{Required Carriageway Lanes } (N) = \left\lceil \frac{V}{\text{Capacity per Lane at LOS C } (\approx 950\text{ PCU/hr})} \right\rceil$$

---

## Hardware & Edge Deployment

| Subsystem | Specification | Purpose |
| :--- | :--- | :--- |
| **Compute Board** | NVIDIA Jetson Nano Developer Kit (4GB B01) | On-device model inference & video processing |
| **GPU** | 128-core NVIDIA Maxwell™ architecture | CUDA / TensorRT hardware acceleration |
| **Camera Module** | Sony IMX219 / IMX477 CSI Camera | High-framerate roadside video capture |
| **Ingestion Pipeline** | Accelerated GStreamer via `nvarguscamerasrc` | Zero-copy frame delivery directly into GPU memory |
| **Quantization** | FP16 Half-Precision Engine | Maintains $>20\text{ FPS}$ while mitigating thermal throttling |

---

## Directory Layout

```
atmas-traffic-monitoring/
├── assets/
│   ├── system_architecture.png
│   └── dashboard_sample.png
├── configs/
│   ├── homography_calibration.json
│   └── road_design_thresholds.yaml
├── models/
│   └── yolov4-tiny.engine
├── src/
│   ├── __init__.py
│   ├── capture.py              # GStreamer CSI & video stream wrappers
│   ├── homography.py           # Planar perspective transform engine
│   ├── kinematics.py           # Velocity, acceleration & trajectory module
│   ├── tracker.py              # Multi-object detection & data association
│   └── civil_engine.py         # IRC:106, IRC:86, IRC:73, and IRC:67 engine
├── ATMAS_Unified_Pipeline.ipynb # Complete interactive Google Colab notebook
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- NVIDIA JetPack SDK (if deploying on Jetson Nano) or CUDA-enabled GPU workstation
- OpenCV with GStreamer support

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/atmas-traffic-monitoring.git
   cd atmas-traffic-monitoring
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Quickstart with Google Colab
Run the unified demonstration notebook with sample video footage and automated highway engineering reporting:
1. Open [`ATMAS_Unified_Pipeline.ipynb`](./ATMAS_Unified_Pipeline.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Run all cells sequentially.
3. Inspect `atmas_telemetry_report.csv` and the generated 4-quadrant engineering diagnostic dashboard.

---

## Telemetry Schema

The pipeline exports a structured CSV report (`atmas_telemetry_report.csv`) with the following fields:

| Field | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| `frame` | Integer | - | Monotonically increasing video frame counter |
| `timestamp_sec` | Float | Seconds | Elapsed observation time |
| `track_id` | Integer | - | Unique persistent identifier assigned to the vehicle |
| `class` | String | - | Detected category (`car`, `motorcycle`, `bus`, `truck`) |
| `speed_kmh` | Float | km/h | Calibrated real-world ground speed |
| `acceleration_mps2`| Float | $m/s^2$ | Instantaneous longitudinal acceleration |
| `pos_x_m` | Float | Meters | Lateral road coordinate relative to ground origin |
| `pos_y_m` | Float | Meters | Longitudinal road coordinate along the monitored section |
| `is_rash` | Boolean | - | Anomaly flag for dangerous driving behavior |
| `reason` | String | - | Anomaly justification (e.g., `Overspeed`, `Harsh Acc/Brake`) |

---

## Patent Information

This technology and its foundational method of edge-based kinematic traffic analysis are protected under the Office of the Controller General of Patents, Designs & Trade Marks (India):

- **Patent Grant Number:** `549991`
- **Application Number:** `202221057506`
- **Filing Date:** November 2022
- **Title:** Autonomous Traffic Monitoring and Analysis System (ATMAS)
- **Inventors:** Dr. Udit Jain, Vaibhav Gidde, et al.
- **Assignee / Institution:** Visvesvaraya National Institute of Technology (VNIT), Nagpur

---

## Citation & Contact

If you use ATMAS in your academic research or industrial benchmarking, please cite:

```bibtex
@patent{ATMAS2024,
  title     = {Autonomous Traffic Monitoring and Analysis System (ATMAS)},
  author    = {Jain, Udit and Gidde, Vaibhav and collaborators},
  number    = {549991},
  type      = {Patent},
  nationality = {Indian},
  year      = {2024},
  note      = {Application No. 202221057506}
}
```

For inquiries regarding commercial deployment, research collaborations, or implementation details:
- **Lead Researcher:** Vaibhav Gidde
- **Academic Advisor:** Dr. Udit Jain (Department of Civil Engineering, VNIT Nagpur)