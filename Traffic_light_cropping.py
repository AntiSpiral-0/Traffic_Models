from ultralytics import YOLO
import cv2
import os

yolo_model = YOLO("yolo11n.pt")

def crop_and_organize_traffic_lights():
    """Crop traffic light sections - ONLY the basic bounding box"""
    result_folders = ['results_green', 'results_red', 'results_yellow']
    for folder in result_folders:
        os.makedirs(folder, exist_ok=True)
    
    folders = ['green', 'red', 'yellow']
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"❌ Folder '{folder}' not found")
            continue
            
        print(f"\n🔍 Processing {folder} traffic lights:")
        
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder, filename)
                
                # Read image
                original_image = cv2.imread(image_path)
                if original_image is None:
                    continue
                
                # YOLO detection
                results = yolo_model(image_path)
                
                # Process detections
                if results[0].boxes is not None:
                    for i, box in enumerate(results[0].boxes):
                        if results[0].names[int(box.cls)] == "traffic light":
                            
                            # Get coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # SINGLE CROP - just the bounding box
                            cropped = original_image[y1:y2, x1:x2]
                            
                            # SINGLE SAVE - one file per detection
                            save_name = f"{filename.split('.')[0]}_crop_{i}.jpg"
                            save_path = os.path.join(f"results_{folder}", save_name)
                            cv2.imwrite(save_path, cropped)
                            
                            print(f"  ✅ Saved: {save_path}")

if __name__ == "__main__":
    crop_and_organize_traffic_lights()