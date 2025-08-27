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

The mask filters out everything outside the ROI.

Only the defined zone remains active for detection.

3. Run YOLOv8 Inside ROI

Vehicles are detected and counted only in the chosen region.

Anything outside the ROI is automatically ignored.

✅ Example Use Cases

🚦 Lane-based vehicle counting.

🅿️ Parking lot occupancy monitoring.

🛂 Toll booth traffic analysis.
