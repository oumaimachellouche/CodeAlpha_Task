🚗 Lane Detection for Autonomous Vehicles

📌 Project Overview

This project focuses on lane detection for autonomous driving systems using classical computer vision techniques. The system detects road lane markings from both images and videos captured by a front-facing camera. The goal is to demonstrate how AI-based perception can be applied to autonomous vehicles in a simple and interpretable way.

🧠 Techniques Used

Grayscale conversion

Gaussian blur for noise reduction

Canny edge detection

Region of Interest (ROI) masking

Hough Transform for lane line detection


🛠️ Tools & Technologies

Python 3.x

OpenCV

NumPy


the project structure

lane_detection/

│── images/                # Input images

│── videos/                # Input videos

│── output/                # Output images with detected lanes

│── output_video/          # Output videos with detected lanes

│── main.py                # Lane detection for images

│── README.md

▶️ How to Run the Project

1️⃣ Install dependencies

pip install opencv-python numpy 

2️⃣ Run lane detection on images

python main.py

3️⃣ Run lane detection on videos

python video_lane_detection.py
