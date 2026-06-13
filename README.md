# Image Quality Simulator (QualityEngine)

A lightweight, robust image processing web application built with FastAPI and OpenCV. This project simulates various digital video release tiers (from low-bitrate CAMRips and VHS degradation to high-quality 4K Remuxes) by performing real-time image manipulation, degradation, and AI-driven upscaling.

## Core Features

- **Exhaustive Quality Simulation:** Supports over 20+ distinct scene release tiers. The engine mathematically simulates the visual artifacts associated with each format:
  - **Low Quality (CAMRip, HDTS, Telesync):** Introduces severe Gaussian blur, high variance noise, and poor contrast.
  - **Standard Definition (VHSRip, Workprint, PDTV):** Simulates chromatic aberration (color bleed), scanlines, random static, desaturation, burnt-in timecodes, and interlacing artifacts.
  - **High Definition (WEB-DL, BDRip, YIFY):** Applies specific bilateral filtering and targeted JPEG compression ratios to mimic various encoding standards.
  - **Ultra High Definition (UHD Remux):** Utilizes EDSR (Enhanced Deep Super-Resolution) via OpenCV's DNN SuperRes module to upscale images seamlessly without losing detail.
- **Interactive UI:** Features a minimal, javascript-driven frontend with an overlaid image comparison slider, allowing users to analyze the "Before and After" of the applied effects in real-time.
- **Fault-Tolerant Engine:** The backend is designed for constraint environments (like Render free tiers). It includes aggressive Python garbage collection, limited OpenCV thread allocation to prevent memory spikes, and comprehensive `try/except` fallbacks that guarantee an image is always returned even if a specific matrix transformation fails.

## Project Structure

```text
.
├── app/
│   ├── main.py            # FastAPI routing, request validation, and endpoint logic
│   └── core/
│       └── engine.py      # QualityEngine class handling all OpenCV manipulation logic
├── static/
│   └── index.html         # Frontend interface and client-side logic
├── EDSR_x4.pb             # Pre-trained AI upscaling model
└── requirements.txt       # Project dependencies
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- The `EDSR_x4.pb` model file must be present in the root directory for the `remux` upscaling tiers to function.

### Installation

1. Clone the repository and navigate to the project root.
2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Once the server is running, navigate to `http://localhost:8000` in your web browser to access the interface.

## Usage Guide

1. **Upload an Image:** Use the file input to upload a custom local image or select one of the provided sample images.
2. **Select Release Tier:** Choose a target degradation or enhancement tier from the dropdown menu (e.g., `VHSRip`, `Workprint`, `UHD Remux`).
3. **Select Resolution (Optional):** Choose an output resolution constraint. If "None" is selected, the application will retain the source resolution (unless overridden by the specific tier logic).
4. **Process:** Click "Upscale Image". The backend will process the image array in memory and return a JPEG binary.
5. **Analyze & Download:** Use the slider in the browser to compare the processed image against the original, and click "Download Image" to save the result.
