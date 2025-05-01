import torch
import cv2
import numpy as np
import easyocr  # Import EasyOCR
import json
from datetime import datetime

# Load YOLOv5 model (using 'yolov5s')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Video source (0 for webcam or provide a video file path)
cap = cv2.VideoCapture("cars.mp4")  # Replace "cars.mp4" with 0 for real-time camera feed

# Initialize video writer to save the output video in MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
output_video = cv2.VideoWriter('output_cars_with_plates.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))  # 20.0 FPS and original resolution

# Initialize vehicle counter and tracking dictionary
vehicle_count = 0
vehicle_trackers = {}  # To track vehicle centroids

# JSON file to store detected vehicle information
vehicle_data = []

# Initialize EasyOCR Reader for English
reader = easyocr.Reader(['en'])  # For English license plates

# Define region of interest (ROI) for vehicle counting
def define_roi(frame):
    height, width, _ = frame.shape
    return (0, int(height * 0.7)), (width, int(height * 0.7))  # ROI at 70% of the frame height

# Function to extract number plate using EasyOCR
def extract_number_plate(frame, x_min, y_min, x_max, y_max):
    # Crop the region where the number plate is expected
    cropped_image = frame[y_min:y_max, x_min:x_max]

    # Convert cropped image to grayscale
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    # Use EasyOCR to extract text from the cropped region
    result = reader.readtext(gray_image)

    # Check if EasyOCR returned a result
    if len(result) > 0:
        number_plate_text = result[0][-2]  # Extract the text from the result
    else:
        number_plate_text = "N/A"

    return number_plate_text.strip()

# Function to process each frame and detect vehicles
def process_frame(frame):
    global vehicle_count, vehicle_trackers

    # Use YOLOv5 for inference
    results = model(frame)

    # Extract detection results: bounding boxes, confidence scores, class labels
    detections = results.pandas().xyxy[0]

    # Dictionary to keep track of the new centroids
    current_centroids = {}

    # Loop over each detection
    for index, detection in detections.iterrows():
        class_name = detection['name']  # Class name (e.g., 'car', 'truck')
        confidence = detection['confidence']
        x_min, y_min, x_max, y_max = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

        # Only consider vehicle classes (e.g., 'car', 'truck', 'bus')
        if class_name in ['car', 'truck', 'bus'] and confidence > 0.5:
            # Draw bounding box around the vehicle
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name}: {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # Get the centroid (center) of the vehicle
            vehicle_center_x = x_min + (x_max - x_min) // 2
            vehicle_center_y = y_min + (y_max - y_min) // 2

            # Assign a unique vehicle ID based on its detection index
            current_centroids[index] = (vehicle_center_x, vehicle_center_y)

            # Define the ROI (toll counting line)
            roi_top, roi_bottom = define_roi(frame)

            # Check if the vehicle crossed the ROI (from top to bottom)
            if index in vehicle_trackers:
                previous_y = vehicle_trackers[index]['centroid'][1]

                # Check if the vehicle is moving from top to bottom (towards the camera) across the ROI
                if previous_y < roi_top[1] and vehicle_center_y >= roi_top[1]:
                    if not vehicle_trackers[index]['counted']:
                        vehicle_count += 1
                        vehicle_trackers[index]['counted'] = True  # Mark this vehicle as counted

                        # Extract number plate and log it
                        number_plate = extract_number_plate(frame, x_min, y_min, x_max, y_max)
                        print(f"Vehicle counted! Total count: {vehicle_count} | Number Plate: {number_plate}")

                        # Append vehicle data to the JSON object
                        vehicle_data.append({
                            "vehicle_id": index,
                            "vehicle_type": class_name,
                            "number_plate": number_plate,
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })

            # Update the tracking information for the current vehicle
            vehicle_trackers[index] = {'centroid': (vehicle_center_x, vehicle_center_y), 'counted': vehicle_trackers.get(index, {}).get('counted', False)}

    # Remove old vehicles that are no longer detected in the frame
    for tracked_vehicle in list(vehicle_trackers.keys()):
        if tracked_vehicle not in current_centroids:
            del vehicle_trackers[tracked_vehicle]

    return frame

# Main loop to process video frames
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Process each frame to detect and count vehicles
    processed_frame = process_frame(frame)

    # Define and draw the ROI line for vehicle counting
    roi_top, roi_bottom = define_roi(frame)
    cv2.line(processed_frame, roi_top, roi_bottom, (0, 0, 255), 2)

    # Display vehicle count
    cv2.putText(processed_frame, f"Vehicle Count: {vehicle_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Write the processed frame into the output video
    output_video.write(processed_frame)

    # Press 'q' to stop the process early if needed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_video.release()
cv2.destroyAllWindows()

# Save vehicle data to JSON file
with open('vehicle_data.json', 'w') as f:
    json.dump(vehicle_data, f, indent=4)

print("Vehicle data saved to 'vehicle_data.json'")
