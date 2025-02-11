# 🚦 Real-Time Vehicle Detection and Tracking using YOLOv8 & SORT

## 📖 Project Overview
This project implements **real-time vehicle detection and tracking** using **YOLOv8 (You Only Look Once) and SORT (Simple Online and Realtime Tracking)**. The system is designed to:
- **Detect** vehicles in real-time from video feeds.
- **Track** detected vehicles across frames while maintaining unique identities.
- **Estimate traffic density and speed** for urban planning and traffic management.

This research is part of **Telecommunication Engineering at Politecnico di Milano** and aims to improve **real-time traffic surveillance and analysis**.

## 🔍 Key Features
✅ **Real-time Vehicle Detection** using YOLOv8  
✅ **Multi-Object Tracking (MOT)** with SORT Algorithm  
✅ **Traffic Density Estimation** for congestion analysis  
✅ **Vehicle Trajectory Visualization** using homography mapping  
✅ **Speed Estimation** based on real-world distances  

## 📁 Project Structure
Track_vehicles.ipynb # Jupyter notebook for vehicle tracking 
real-time-vehicles-detection.ipynb # Notebook for real-time detection 
output_samples # Video samples for showing model's outcome.
runs # Yolo reports for training.
sample # Video samples.
sort # Sort tracking algorithm


## 🎯 Project Objectives
1. **Implement YOLOv8** for real-time vehicle detection.
2. **Use SORT for multi-object tracking**, ensuring each detected vehicle is consistently identified across frames.
3. **Estimate traffic density** by counting vehicles in defined areas.
4. **Calculate vehicle speed** using homography mapping and frame time differences.
5. **Visualize vehicle trajectories** for better traffic flow analysis.


### Install Dependencies
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/your-username/Real-Time-Vehicle-Detection.git
cd Real-Time-Vehicle-Detection
pip install -r sort/requirements.txt
pip install jupyter 
```
other requeirments are in the jupyter files

## 📊 Methodology

### 1️⃣ **Object Detection (YOLOv8)**
- YOLOv8 is a deep-learning-based **real-time object detector**.
- The model was trained on a **vehicle-specific dataset**.
- Bounding boxes are generated for each detected vehicle.

### 2️⃣ **Object Tracking (SORT Algorithm)**
- **Kalman filter** predicts vehicle movement.
- **Hungarian algorithm** matches new detections with existing tracks.
- SORT ensures each vehicle maintains a unique ID across frames.

### 3️⃣ **Traffic Density Estimation**
- The system **counts vehicles per frame** in predefined regions.
- Helps in **real-time congestion analysis**.

### 4️⃣ **Speed Estimation using Homography**
- Converts image coordinates to **real-world distances**.
- Uses frame timestamps to compute vehicle **speed in km/h**.
- Applies **moving average filters** for smoother speed estimation.

## 🎥 Output Results

### 🚗 Vehicle Detection  
[![Vehicle Detection](output_samples/sample_image.jpg)](output_samples/output_video.mp4 "Click to view the video")  

### 🚘 Vehicle Tracking  
[![Vehicle Tracking](output_samples/sample_image.jpg)](output_samples/processed_sample_video.mp4 "Click to view the video")  

### 📍 Trajectory Mapping  
[![Trajectory Mapping](output_samples/sample_image.jpg)](output_samples/output_video_with_trajectory.mp4 "Click to view the video")  

### 🚦 Traffic Density Estimation  
[![Traffic Density](output_samples/sample_image.jpg)](output_samples/traffic_density_analysis.mp4 "Click to view the video")  

---

## 📈 Model Performance

| Metric  | Value |
|---------|-------|
| **Precision** | 88% |
| **Recall** | 84% |
| **mAP (Mean Average Precision)** | 82% |
| **FPS (Processing Speed)** | 30+ FPS |

---

## 💡 Future Improvements

### 🔹 **Enhancing Tracking Robustness**
- Improve object tracking accuracy in **high-density traffic** where vehicles occlude each other.
- Implement **DeepSORT** for a more advanced tracking approach.

### 🔹 **Improving Speed Estimation Accuracy**
- Fine-tune **homography mapping** to ensure better real-world distance estimation.
- Implement a **Kalman filter-based smoothing technique** to refine speed calculations.

### 🔹 **Deploying the System on Edge Devices**
- Optimize the model for **real-time deployment** on **Raspberry Pi** and **NVIDIA Jetson**.
- Reduce **compute requirements** while maintaining detection accuracy.

### 🔹 **Integrating Traffic Light Optimization**
- Develop a **smart traffic light control system** based on real-time congestion data.
- Use **Reinforcement Learning (RL)** to adapt traffic signals dynamically.
