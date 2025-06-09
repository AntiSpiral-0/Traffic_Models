from ultralytics import YOLO, SAM
import cv2
import os

# Load models
yolo_model = YOLO("yolo11n.pt")
sam_model = SAM("sam2.1_b.pt")

# Open video
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('car_segmented_video.mp4', fourcc, fps, (width, height))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Save frame temporarily
    temp_frame_path = f"temp_frame_{frame_count}.jpg"
    cv2.imwrite(temp_frame_path, frame)
    
    # Detect cars with YOLO
    yolo_results = yolo_model(temp_frame_path)
    
    # Process each detected car
    for box in yolo_results[0].boxes:
        if yolo_results[0].names[int(box.cls)] == "car":
            bbox = box.xyxy[0].tolist()
            
            # Segment with SAM
            sam_results = sam_model(temp_frame_path, bboxes=[bbox])
            annotated_frame = sam_results[0].plot()
            
            # Write to output video
            out.write(annotated_frame)
            break
    else:
        # No car detected, write original frame
        out.write(frame)
    
    # Clean up temp file
    os.remove(temp_frame_path)
    frame_count += 1
    
    if frame_count % 30 == 0:  # Progress indicator
        print(f"Processed {frame_count} frames")

cap.release()
out.release()
print("Video processing complete!")