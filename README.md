# Advanced Surveillance System

This project detects and tracks people in real time using YOLOv8 and Norfair.  
It also simulates servo movements to follow the target and blurs non-tracked faces.  
Optimized for Raspberry Pi integration with servo motors.

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
git clone https://github.com/<your-username>/Advanced-Surveillance-System.git
cd Advanced-Surveillance-System
2. Create Virtual Environment
bash
Copy code
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # Linux/Mac
3. Install Packages
bash
Copy code
pip install -r requirements.txt
4. Run the Project
Add a test video in videos/ folder
or change to webcam in main.py:

python
Copy code
cap = cv2.VideoCapture(0)
Then run:

bash
Copy code
python main.py
Press Q to quit.

🧩 Folder Overview
bash
Copy code
main.py                # Main loop
detection/detector.py  # YOLOv8 detection
tracking/tracker.py    # Multi-person tracking
control/servo_controller.py  # Servo movement simulation
privacy/face_blur.py   # Face blurring for others
