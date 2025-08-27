# object_detection/common_modules/vehicle_detector_class.py

from ultralytics import YOLO
import cv2
import cvzone
import torch
import numpy as np
import os

# Assuming sort.py is also in common_modules/
from sort.sort import Sort


class countVehicles:
    """
    A class to perform vehicle detection and counting on a video stream
    using YOLOv8 and SORT tracking.
    """

    def __init__(self, model_path='../Yolo-Weights/yolov8l=.pt', mask_path=None):
        """
        Initializes the VehicleDetector with a YOLO model and an optional mask.

        Args:
            model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8l.pt').
                              Relative path assumes it's relative to this script's location.
            mask_path (str, optional): Path to a binary mask image (PNG, JPG).
                                      White areas allow detection, black areas block.
                                      If None, no mask is applied.
        """
        # --- Load YOLOv8 Model ---
        self.model = YOLO(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"YOLOv8 model loaded successfully on {self.device}.")

        # --- COCO class names (hardcoded as per your snippet) ---
        self.classNames = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "bird",
            "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
            "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
            "scissors", "teddy bear", "hair drier", "toothbrush", "fish", "frog", "snake", "tiger", "rabbit",
            "lion", "panda", "kangaroo", "camel", "goat", "monkey", "penguin", "parrot", "owl", "bat",
            "shark", "whale", "dolphin", "lobster", "crab", "butterfly", "spider", "ant", "bee", "snail", "Mobile",
            "Keyboard", "Mouse", "Bottle", "charger", "Medicine bottle", "Rubber Band", "Television", "Chair", "remote"
        ]

        # Vehicle classes only
        self.allowed_class_ids = [2, 3, 5, 7]  # car, motorbike, bus, truck

        # --- Load Mask if available ---
        self.mask = None
        if mask_path:
            absolute_mask_path = os.path.abspath(mask_path)
            if os.path.exists(absolute_mask_path):
                self.mask = cv2.imread(absolute_mask_path)
                if self.mask is None:
                    print(
                        f"Warning: Mask file '{absolute_mask_path}' could not be loaded. Check if it's a valid image.")
            else:
                print(f"Warning: Mask file not found at '{absolute_mask_path}'. Proceeding without mask.")
        else:
            print("No mask path provided. Proceeding without mask.")

        # --- Tracker ---
        self.tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

        # --- Counted IDs ---
        self.totalCount = []  # List to store unique IDs of counted vehicles

    def detect_from_video(self, video_path):
        """
        Processes a video file to perform vehicle detection and counting.

        Args:
            video_path (str): Path to the video file.
        """
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video file: {video_path}")
            return

        # Pre-read first frame to get video dimensions for mask resizing
        ret, temp_frame = cap.read()
        if not ret:
            print("Could not read first frame to determine video dimensions.")
            cap.release()
            return
        video_width = temp_frame.shape[1]
        video_height = temp_frame.shape[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

        # If mask is loaded, resize it once to video dimensions
        if self.mask is not None:
            if self.mask.shape[:2] != (video_height, video_width):
                self.mask = cv2.resize(self.mask, (video_width, video_height))
            print(f"Mask loaded and resized to video dimensions: {self.mask.shape[:2]}")

        print(f"Starting video processing: {video_path}")
        while True:
            success, img = cap.read()
            if not success:
                print("✅ Video ended or error in reading frame.")
                break

            # Apply mask to the current frame if available
            if self.mask is not None:
                imgRegion = cv2.bitwise_and(img, self.mask)
            else:
                imgRegion = img.copy()  # Use the original image if no mask

            # Perform YOLO detection on the (potentially masked) image region
            results = self.model(imgRegion, stream=True)
            detections = np.empty((0, 5))
            detection_info = []  # To store [x1, y1, x2, y2, class_name]

            for r in results:
                for box in r.boxes:
                    if hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy'):
                        cls = int(box.cls[0])
                        if cls not in self.allowed_class_ids:
                            continue
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Accumulate detections for SORT
                        detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

                        # Store info for matching later with tracker results
                        class_name = self.classNames[cls]
                        detection_info.append([x1, y1, x2, y2, class_name])

            # Update the SORT tracker with current detections
            resultsTracker = self.tracker.update(detections)

            # Process tracked objects
            for result in resultsTracker:
                x1, y1, x2, y2, id = map(int, result)
                w, h = x2 - x1, y2 - y1
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Match with detection_info by IoU to get class name
                max_iou = 0
                matched_class = "object"  # Default if no match found
                for dx1, dy1, dx2, dy2, class_name in detection_info:
                    # IoU calculation to associate tracker ID with its class name
                    xi1 = max(x1, dx1)
                    yi1 = max(y1, dy1)
                    xi2 = min(x2, dx2)
                    yi2 = min(y2, dy2)
                    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                    box_area = (x2 - x1) * (y2 - y1)
                    det_area = (dx2 - dx1) * (dy2 - dy1)
                    union_area = box_area + det_area - inter_area
                    iou = inter_area / union_area if union_area != 0 else 0

                    if iou > max_iou:
                        max_iou = iou
                        matched_class = class_name

                label = f"{matched_class} {int(id)}"

                # Draw bounding box and label
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
                cvzone.putTextRect(
                    img, label, (max(0, x1), max(35, y1 - 10)),
                    scale=2, thickness=3,
                    colorT=(255, 255, 255),
                    colorR=(170, 0, 200),
                    border=2, offset=10
                )
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                # Simple counting logic (as per your working code)
                # Note: For more robust counting (e.g., preventing double counts or counting on line crossing),
                #       you'd implement the advanced logic discussed previously (previous_positions, counted_ids).
                if id not in self.totalCount:
                    self.totalCount.append(id)

            # Display current total count
            cvzone.putTextRect(img, f'Count: {len(self.totalCount)}', (50, 50), scale=2, thickness=2)

            # Display the video frame
            cv2.imshow("Vehicle Detection", img)

            # Exit on 'q' press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Release video capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")


# --- Example of how to use this class directly (for testing this module) ---
if __name__ == "__main__":
    # Define paths relative to the current script's location for this direct run
    # (These paths assume a structure like: object_detection/common_modules/vehicle_detector_class.py)
    # The '..' means go up one directory from 'common_modules' to 'object_detection'
    # The '../Yolo-Weights/yolov8l.pt' path is correct from common_modules/ to Yolo-Weights/

    # Path to YOLO model weights (relative to vehicle_detector_class.py)
    model_weights = '../Yolo-Weights/yolov8l.pt'

    # Path to the mask image (relative to vehicle_detector_class.py or absolute)
    # If Mask.png is in Car_Counter, you'd need to adjust this path to go up two folders, then into Car_Counter
    # Example: mask_image = '../Car_Counter/Mask.png'
    # Or provide an absolute path: mask_image = 'D:/Object detection/Car_Counter/Mask.png'
    mask_image = '../Car_Counter/Mask.png'  # Assuming Mask.png is in Car_Counter

    # Path to the video file (relative to vehicle_detector_class.py or absolute)
    # Example: video_file = '../../Videos/Traffic.mp4'
    # Or provide an absolute path: video_file = 'D:/Object detection/Videos/Traffic.mp4'
    video_file = '../../Videos/Traffic.mp4'  # Assuming Videos folder is sibling to Car_Counter/number_Recognition

    print(f"Attempting to load model from: {os.path.abspath(model_weights)}")
    print(f"Attempting to load mask from: {os.path.abspath(mask_image) if mask_image else 'None'}")
    print(f"Attempting to process video: {os.path.abspath(video_file)}")

    # Create an instance of the VehicleDetector class
    detector = countVehicles(model_path=model_weights, mask_path=mask_image)

    # Process the video
    detector.detect_from_video(video_path=video_file)
