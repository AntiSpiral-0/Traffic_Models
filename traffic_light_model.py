"""
TRAFFIC LIGHT COLOR DETECTION MODEL

This file uses the cropped traffic light images from Traffic_light_cropping.py
to train a color classification model using RGB pixel analysis.

Input: results_green/, results_red/, results_yellow/ folders (cropped traffic lights)
Output: Color classification model and accuracy results
"""

import cv2
import os
import numpy as np
import shutil

def analyze_traffic_light_color(image_path):
    """
    Analyze cropped traffic light image to determine color using RGB pixel analysis
    Your idea: Use RGB pixels to find dominant color in condensed regions
    """
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        return "unknown", {}
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Define color ranges in RGB - these detect concentrated color regions
    color_ranges = {
        'red': {
            'lower': np.array([100, 0, 0]),      
            'upper': np.array([255, 120, 120])
        },
        'yellow': {
            'lower': np.array([50, 50, 0]),    
            'upper': np.array([255, 255, 200])
        },
        'green': {
            'lower': np.array([0, 50, 0]),      
            'upper': np.array([200, 255, 200])
        }
    }
    
    color_scores = {}
    
    for color_name, ranges in color_ranges.items():
        # Create mask for this color range
        mask = cv2.inRange(image_rgb, ranges['lower'], ranges['upper'])
        
        # Count pixels in this color range
        color_pixels = cv2.countNonZero(mask)
        
        # Calculate density (pixels per total area)
        total_pixels = image.shape[0] * image.shape[1]
        density = color_pixels / total_pixels
        
        color_scores[color_name] = {
            'pixels': color_pixels,
            'density': density,
            'percentage': (color_pixels / total_pixels) * 100
        }
    
    # Find dominant color - the one with highest density of concentrated pixels
    dominant_color = max(color_scores.keys(), key=lambda k: color_scores[k]['density'])
    
    # Minimum threshold to avoid false positives (ignore if too few colored pixels)
    min_threshold = 0.05  # 5% of pixels must be the target color
    
    if color_scores[dominant_color]['density'] < min_threshold:
        return "off", color_scores
    
    return dominant_color, color_scores

def train_rgb_color_classifier():
    """
    Train and test the RGB pixel analysis classifier on cropped traffic lights
    """
    
    # Check if cropped folders exist
    result_folders = ['results_green', 'results_red', 'results_yellow']
    
    # Create classification result folders
    classification_folders = ['classified_red', 'classified_yellow', 'classified_green', 'classified_off']
    for folder in classification_folders:
        os.makedirs(folder, exist_ok=True)
    
    total_classified = 0
    correct_classifications = 0
    classification_report = {}
    
    print("🚦 TRAFFIC LIGHT COLOR CLASSIFICATION TRAINING")
    print("="*60)
    print("Method: RGB Pixel Analysis (Your Idea)")
    print("Logic: Find concentrated color regions in cropped traffic lights")
    print("="*60)
    
    for folder in result_folders:
        if not os.path.exists(folder):
            print(f"❌ Folder '{folder}' not found")
            print(f"   Run Traffic_light_cropping.py first to create cropped images")
            continue
            
        print(f"\n🔍 Classifying images from {folder}:")
        expected_color = folder.split('_')[1]  # Extract expected color from folder name
        
        folder_total = 0
        folder_correct = 0
        
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder, filename)
                
                # Analyze color using RGB pixel analysis
                detected_color, scores = analyze_traffic_light_color(image_path)
                
                # Move to appropriate classification folder
                dest_folder = f"classified_{detected_color}"
                dest_path = os.path.join(dest_folder, f"{expected_color}_{filename}")
                
                # Copy image to classification folder
                shutil.copy2(image_path, dest_path)
                
                # Check if classification is correct
                is_correct = detected_color == expected_color
                if is_correct:
                    correct_classifications += 1
                    folder_correct += 1
                
                total_classified += 1
                folder_total += 1
                
                status = "✅" if is_correct else "❌"
                print(f"  {status} {filename}: Expected {expected_color}, Got {detected_color}")
                
                # Show pixel analysis details
                for color, score in scores.items():
                    print(f"    {color}: {score['percentage']:.1f}% ({score['pixels']} pixels)")
        
        # Store folder results
        folder_accuracy = (folder_correct / folder_total * 100) if folder_total > 0 else 0
        classification_report[expected_color] = {
            'correct': folder_correct,
            'total': folder_total,
            'accuracy': folder_accuracy
        }
        
        print(f"📊 {expected_color} accuracy: {folder_correct}/{folder_total} ({folder_accuracy:.1f}%)")
    
    # Calculate overall accuracy
    overall_accuracy = (correct_classifications / total_classified * 100) if total_classified > 0 else 0
    
    print(f"\n" + "="*60)
    print(f"🎯 FINAL RESULTS")
    print(f"="*60)
    print(f"Overall Accuracy: {correct_classifications}/{total_classified} ({overall_accuracy:.1f}%)")
    print(f"\nPer-color breakdown:")
    for color, stats in classification_report.items():
        print(f"  {color.upper()}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
    # Show classification results
    print(f"\n📁 Classification results saved to:")
    for folder in classification_folders:
        if os.path.exists(folder):
            count = len([f for f in os.listdir(folder) if f.endswith('.jpg')])
            print(f"  {folder}/: {count} images")
    
    # Performance analysis
    print(f"\n🔬 ANALYSIS:")
    if overall_accuracy >= 80:
        print(f"✅ Excellent performance! RGB pixel analysis works well for your dataset.")
    elif overall_accuracy >= 60:
        print(f"⚠️  Good performance. Consider tuning color ranges for better results.")
    else:
        print(f"❌ Poor performance. May need to adjust color thresholds or try HSV analysis.")
    
    return overall_accuracy, classification_report

def test_on_new_image(image_path):
    """
    Test the trained RGB classifier on a new cropped traffic light image
    """
    print(f"\n🧪 Testing on new image: {image_path}")
    
    detected_color, scores = analyze_traffic_light_color(image_path)
    
    print(f"🎯 Detected color: {detected_color.upper()}")
    print(f"📊 Color analysis:")
    for color, score in scores.items():
        print(f"  {color}: {score['percentage']:.1f}%")
    
    return detected_color

# Main execution
if __name__ == "__main__":
    print("Starting RGB-based traffic light color classification...")
    
    # Train and test the classifier
    accuracy, report = train_rgb_color_classifier()
    
    # Optional: Test on a specific image
    # test_image = "results_red/some_image.jpg"  # Replace with actual image path
    # if os.path.exists(test_image):
    #     test_on_new_image(test_image)
    
    print(f"\n🎉 Training complete! Model accuracy: {accuracy:.1f}%")