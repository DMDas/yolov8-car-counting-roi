import sys
import os

# Add the parent directory (D:\Object detection) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the class
from Car_Counter.Car_Counter import countVehicles

if __name__ == "__main__":
    # Define paths
    model_weights = "D:/Object detection/Yolo-Weights/yolov8l.pt"
    mask_path = "D:/Object detection/Car_Counter/Mask.png"
    video_path = "D:/Object detection/Videos/Traffic.mp4"

    # Create the object with mask and model path
    vehicle_counter = countVehicles(model_path=model_weights, mask_path=mask_path)

    # Process the video with mask applied
    vehicle_counter.detect_from_video(video_path)