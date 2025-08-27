YOLOv8 Car Counting with ROI Mask

Detect and count vehicles only inside your chosen Region of Interest (ROI) using YOLOv8 + OpenCV masking. This ensures vehicles outside the zone are ignored — ideal for lane-wise monitoring, parking zones, or toll plazas.

✨ Key Features

🎯 ROI-restricted detection & counting – ignore vehicles outside your zone.

🖼️ Image-based mask – design the ROI in Canva (or any editor).

⚡ Plug-and-play with YOLOv8 – works with all Ultralytics model sizes.

📹 Works on videos & live streams – real-time or offline.

🔍 Simple pipeline – fast, accurate, and easy to reproduce.

🧠 How It Works
1. Prepare the ROI Mask

Take a frame/screenshot of your video to get the exact resolution.

In Canva (or any editor):

Cover unwanted regions with black rectangles.

Keep only the detection zone in white.

Export as a .png mask (must match video resolution).

▶️ Watch this short demo video to see how to create the mask:
[Insert your video here]

2. Apply the Mask per Frame

The mask is applied using cv2.bitwise_and(), which keeps only the pixels inside your ROI.

How it works:

Bitwise AND compares each pixel of the video frame with the mask.

A pixel remains visible (colored) only if it is present in both the frame and the white area of the mask.

Pixels outside the ROI (where the mask is black) are turned black (ignored).

This way, only your chosen region of interest is preserved, and everything else is removed before running detection.

3. Run YOLOv8 Inside ROI

YOLOv8 processes only the masked frame.

Vehicles are detected and counted only in the chosen region.

Detections outside are automatically excluded because the pixels were masked away.

✅ Example Use Cases

🚦 Lane-based vehicle counting.

🅿️ Parking lot occupancy monitoring.

🛂 Toll booth traffic analysis.
