from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLO model
yolo_model = YOLO("yolo11n.pt")

def analyze_traffic_light_color(image_crop):
    """
    Analyze cropped traffic light image to determine color using RGB pixel analysis
    """
    
    if image_crop is None or image_crop.size == 0:
        return "unknown", {}
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    
    # Define color ranges in RGB - balanced thresholds
    color_ranges = {
        'red': {
            'lower': np.array([100, 0, 0]),      
            'upper': np.array([255, 120, 120])
        },
        'yellow': {
            'lower': np.array([80, 80, 0]),    
            'upper': np.array([255, 255, 150])
        },
        'green': {
            'lower': np.array([0, 80, 0]),      
            'upper': np.array([120, 255, 120])
        }
    }
    
    color_scores = {}
    
    for color_name, ranges in color_ranges.items():
        # Create mask for this color range
        mask = cv2.inRange(image_rgb, ranges['lower'], ranges['upper'])
        
        # Add special handling for red (since it's problematic)
        if color_name == 'red':
            # Also check for pixels where red channel dominates
            red_channel = image_rgb[:,:,0]
            green_channel = image_rgb[:,:,1]
            blue_channel = image_rgb[:,:,2]
            
            # Red dominance: red > green+20 AND red > blue+20 AND red > 60
            red_dominance = (red_channel > green_channel + 20) & \
                          (red_channel > blue_channel + 20) & \
                          (red_channel > 60)
            red_dominance_mask = red_dominance.astype(np.uint8) * 255
            
            # Combine the masks
            mask = cv2.bitwise_or(mask, red_dominance_mask)
        
        # Count pixels in this color range
        color_pixels = cv2.countNonZero(mask)
        
        # Calculate density (pixels per total area)
        total_pixels = image_crop.shape[0] * image_crop.shape[1]
        density = color_pixels / total_pixels if total_pixels > 0 else 0
        
        color_scores[color_name] = {
            'pixels': color_pixels,
            'density': density,
            'percentage': (color_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        }
    
    # Find dominant color
    dominant_color = max(color_scores.keys(), key=lambda k: color_scores[k]['density'])
    
    # Balanced threshold - 3%
    min_threshold = 0.03
    
    if color_scores[dominant_color]['density'] < min_threshold:
        return "off", color_scores
    
    return dominant_color, color_scores

def traffic_light_detection_poc():
    """
    PoC: Detect traffic lights in image1.png and classify their colors
    """
    
    # Input image
    image_path = "image1.png"
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read {image_path}")
        print("Make sure image1.png exists in the current directory")
        return
    
    print(f"🔍 Analyzing {image_path}...")
    print(f"Image size: {image.shape}")
    
    # Detect traffic lights with YOLO
    results = yolo_model(image_path)
    
    # Create output image (copy of original)
    output_image = image.copy()
    
    traffic_lights_found = 0
    
    if results[0].boxes is not None:
        for i, box in enumerate(results[0].boxes):
            if results[0].names[int(box.cls)] == "traffic light":
                traffic_lights_found += 1
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                confidence = box.conf[0]
                
                print(f"\n🚦 Traffic Light {i+1} detected:")
                print(f"   Location: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"   Confidence: {confidence:.2f}")
                
                # Crop the traffic light region
                traffic_light_crop = image[y1:y2, x1:x2]
                
                # Analyze the color
                detected_color, scores = analyze_traffic_light_color(traffic_light_crop)
                
                print(f"   Detected color: {detected_color.upper()}")
                print(f"   Color analysis:")
                for color, score in scores.items():
                    print(f"     {color}: {score['percentage']:.1f}%")
                
                # Choose box color based on detected traffic light color
                color_map = {
                    'red': (0, 0, 255),      # Red box
                    'yellow': (0, 255, 255), # Yellow box  
                    'green': (0, 255, 0),    # Green box
                    'off': (128, 128, 128)   # Gray box
                }
                
                box_color = color_map.get(detected_color, (255, 255, 255))  # White if unknown
                
                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), box_color, 3)
                
                # Add label with color and confidence
                label = f"{detected_color.upper()} ({confidence:.2f})"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # Draw label background
                cv2.rectangle(output_image, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1), 
                            box_color, -1)
                
                # Draw label text
                cv2.putText(output_image, label, 
                          (x1 + 5, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Save the result
    output_path = "traffic_light_detection_result.jpg"
    cv2.imwrite(output_path, output_image)
    
    # Print summary
    print(f"\n" + "="*50)
    print(f"🎯 TRAFFIC LIGHT DETECTION PoC RESULTS")
    print(f"="*50)
    print(f"Input image: {image_path}")
    print(f"Output image: {output_path}")
    print(f"Traffic lights detected: {traffic_lights_found}")
    
    if traffic_lights_found > 0:
        print(f"✅ Detection successful! Check {output_path}")
    else:
        print(f"❌ No traffic lights detected in the image")
        print(f"💡 Try with an image that contains visible traffic lights")
    
    print(f"\n🔬 Legend:")
    print(f"  🔴 Red box = Red traffic light")
    print(f"  🟡 Yellow box = Yellow traffic light") 
    print(f"  🟢 Green box = Green traffic light")
    print(f"  ⚫ Gray box = Traffic light off/unknown")

if __name__ == "__main__":
    print("🚦 TRAFFIC LIGHT COLOR DETECTION - PROOF OF CONCEPT")
    print("="*60)
    print("This PoC detects traffic lights and classifies their colors")
    print("Input: image1.png")
    print("Output: traffic_light_detection_result.jpg with colored boxes")
    print("="*60)
    
    traffic_light_detection_poc()