# EchoVision 3D - AI-Powered Spatial Audio & Navigation System

📌 **What This Project Does**

EchoVision 3D is a real-time assistive navigation system designed for blind and low-vision individuals. Instead of relying solely on visual feedback or basic text-to-speech, it converts real-time camera video streams into intuitive 3D spatial audio and binaural cues. Users can "hear" the distance, direction, elevation, and identity of obstacles around them in real time.

To ensure real-time reliability and zero-latency feedback, we engineered a hybrid 3-tier detection engine:

1. **Tier 1 (Classical Computer Vision - Edge & Motion):** Always-on, model-free background calibration, optical flow, Sobel edge density, and custom centroid tracking. It operates offline with zero AI inference latency to map immediate obstacles.
2. **Tier 2 (AI Object Identification - Proximity & Bounding):** Real-time 80-class object detection using YOLOv8n to identify key items (e.g., chairs, people, doors) and calculate proximity.
3. **Tier 3 (Scene Description - Natural Language):** On-demand voice-activated AI scene narration using Groq's Vision API (Llama 4 Scout) to provide full contextual descriptions of the environment.

Rather than treating spatial feedback as simple stereo volume, we engineered a 4-dimensional sonification mapping model:

* **Horizontal Angle (Pan):** Managed via Web Audio HRTF (Head-Related Transfer Function) panners to shift sounds precisely across left and right channels.
* **Distance (Proximity):** Mapped dynamically to volume gain and pulse frequency—closer objects sound louder and pulse rapidly.
* **Height (Elevation):** Encoded into audio pitch—ground-level (190 Hz), mid-level (420 Hz), and head-height hazards (820 Hz).
* **Velocity & Collision:** Calculates Time-to-Collision (TTC) using optical flow vectors to generate Doppler-style pitch shifts for fast-approaching threats.

📊 **Results**

We benchmarked EchoVision 3D to ensure it meets real-time assistive hardware demands across frame rate, latency, and system stability:

1. **Real-Time FPS Performance:** Achieved a stable ~10 FPS full-duplex video processing loop over persistent WebSockets without dropping frames or stalling client rendering.
2. **Sub-50ms Audio Response:** Web Audio API node-pooling reduced spatial sound field response latency to under 50ms, delivering instantaneous acoustic feedback as camera angles change.
3. **Zero Heartbeat Timeout Rate:** Asynchronous thread-pooling offloaded PyTorch model passes from the FastAPI main event loop, achieving a 0% WebSocket connection drop rate during heavy AI inference.

🛠 **Tech Stack**

1. **Core Backend Engine:** Python & FastAPI - Handles concurrent full-duplex WebSocket connections for streaming video frames and audio metadata.
2. **Classical Vision Pipeline:** OpenCV - Powers Sobel edge density filtering, Lucas-Kanade optical flow, and frame differencing for zero-latency obstacle mapping.
3. **AI Vision & Object Detection:** PyTorch & YOLOv8 Nano (`yolov8n.pt`) - Lightweight, real-time object detection across 80 COCO classes.
4. **AI Reasoning & Narration Layer:** Groq Vision API (Llama 4 Scout) - Delivers low-latency multi-sentence scene narration on demand.
5. **Frontend & Audio Engine:** HTML5, CSS3, Vanilla JavaScript, Web Audio API (HRTF PannerNodes, GainNodes, Oscillators) - Renders 3D spatialized audio directly in the browser without third-party plugins.
6. **Voice Interface:** Web Speech API - Enables hands-free voice command input and spoken scene description playback.

🚀 **How to Run**

1. **Prerequisites:** Python 3.12 installed on your system along with a webcam or video input device.
2. **API Keys:** Obtain a free API key from [Groq Cloud](https://console.groq.com/) for Tier 3 AI scene narration.
3. Download the folder.
4. Run these commands in your terminal:
   ```bash
   cd backend
   pip install -r requirements.txt
   pip install ultralytics
   py -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
