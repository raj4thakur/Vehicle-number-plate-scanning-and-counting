# 🚗 Vehicle Detection, Counting & License Plate Recognition

This project uses **YOLOv5** and **EasyOCR** to detect, count, and extract number plates from vehicles in a video feed. It draws bounding boxes, counts cars crossing a virtual line, extracts license plates, and logs everything to a JSON file.

---

## 📦 Features

- ✅ Vehicle detection using YOLOv5 (`car`, `truck`, `bus`)
- ✅ Vehicle counting via a virtual Region of Interest (ROI) line
- ✅ Number plate recognition using EasyOCR
- ✅ Logs vehicle type, plate number, and timestamp to `vehicle_data.json`
- ✅ Saves annotated output video as `output_cars_with_plates.mp4`

---

## 🔧 Requirements

Install the following Python packages:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python easyocr numpy
