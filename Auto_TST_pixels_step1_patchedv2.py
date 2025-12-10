import cv2
import numpy as np
import ultralytics
from PIL import Image
import os
import matplotlib.pyplot as plt
import math
from collections import deque
from pathlib import Path

# ---- Similarity toggle ----
SIM_K = 1
SIM_AGG = "median"  # or "mean"

def calculate_circularity(segment):
    area = cv2.contourArea(segment)
    perimeter = cv2.arcLength(segment, True)
    if perimeter == 0:
        return 0, 0
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity, area

def calculate_color_histogram(segment, frame):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [segment], 1)
    hist = cv2.calcHist([frame], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def contour_similarity(prev_cnt, curr_cnt, sigma=0.35):
    if prev_cnt is None or curr_cnt is None:
        return 1.0
    d = cv2.matchShapes(prev_cnt, curr_cnt, cv2.CONTOURS_MATCH_I1, 0.0)
    sim = math.exp(-d / max(sigma, 1e-6))
    return float(np.clip(sim, 0.0, 1.0))

def contour_centroid(cnt):
    if cnt is None:
        return None
    M = cv2.moments(cnt)
    if M["m00"] != 0:
        return (M["m10"]/M["m00"], M["m01"]/M["m00"])
    pts = cnt[:, 0, :].astype(np.float32)
    return (float(pts[:,0].mean()), float(pts[:,1].mean()))

def pick_index_nearest(segments, prev_centroid):
    if prev_centroid is None or not segments:
        return 0 if segments else None
    p = np.array(prev_centroid, dtype=np.float32)
    dmin, imin = None, None
    for i, seg in enumerate(segments):
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2)
        c = pts.mean(axis=0)
        d = np.linalg.norm(c - p)
        if (dmin is None) or (d < dmin):
            dmin, imin = d, i
    return imin

def process_video(video_path, model, output_dir):
    """Process a single video file."""
    video_name = Path(video_path).stem
    image_folder = os.path.join(output_dir, f'{video_name}_images')
    csv_save_path = os.path.join(output_dir, f'{video_name}_data.npy')
    
    print(f"\nProcessing video: {video_name}")
    
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    if not ret:
        print(f"Error: Could not read video {video_path}")
        cap.release()
        return
    cap.release()

    results = model.track(source=video_path, imgsz=640, stream=True, device="cuda:0", persist=True, conf=0.5, max_det=3)

    enclosed_pixels = []
    shape_similarity = []

    os.makedirs(image_folder, exist_ok=True)
    mouse_color_histogram = None
    selected_segment_index = 0
    mouse_selected = False

    prev_contours = deque(maxlen=SIM_K)
    prev_centroid = None

    for index, result in enumerate(results):
        segments = result.masks.xy if (result.masks is not None and hasattr(result.masks, "xy")) else []
        im = Image.fromarray(result.plot())
        image_path = os.path.join(image_folder, str(index) + ".jpg")
        im.save(image_path)
        frame = np.array(im)

        if (index == 0) or (not mouse_selected):
            while not mouse_selected:
                if result.orig_img is not None:
                    plt.imshow(result.orig_img)
                print("Segments detected:")
                for i, segment in enumerate(segments):
                    seg_np = np.array(segment)
                    plt.plot(seg_np[:, 0], seg_np[:, 1], label='Segment {}'.format(i))
                plt.legend()
                plt.show()

                sel = input("Enter the index of the mouse segment (0..{}) or 'skip': ".format(len(segments)-1)).strip().lower()
                if sel == 'skip':
                    break
                try:
                    selected_segment_index = int(sel)
                    assert 0 <= selected_segment_index < len(segments)
                except Exception:
                    print("Invalid selection; defaulting to 0")
                    selected_segment_index = 0

                selected_segment = segments[selected_segment_index]
                segments_array = np.array(selected_segment, np.int32).reshape((-1, 1, 2))
                mouse_color_histogram = calculate_color_histogram(segments_array, frame)
                mouse_selected = True

            if not mouse_selected:
                print("Skipping frame {} as no valid mouse segment was found.".format(index))
                enclosed_pixels.append(np.nan)
                shape_similarity.append(np.nan)
                continue

        if len(segments) == 0:
            enclosed_pixels.append(np.nan)
            shape_similarity.append(np.nan)
            continue

        if not (0 <= selected_segment_index < len(segments)):
            sel_idx = pick_index_nearest(segments, prev_centroid)
        else:
            sel_idx = selected_segment_index

        if sel_idx is None:
            enclosed_pixels.append(np.nan)
            shape_similarity.append(np.nan)
            continue

        segment = segments[sel_idx]
        cnt = np.array(segment, np.int32).reshape((-1, 1, 2))

        height, width = result.orig_shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [cnt], 1)
        area = int(mask.sum())
        enclosed_pixels.append(area)

        if len(prev_contours) == 0:
            sim = 1.0
        else:
            if SIM_K == 1:
                sim = contour_similarity(prev_contours[-1], cnt)
            else:
                sims = [contour_similarity(pc, cnt) for pc in prev_contours]
                sim = float(np.median(sims)) if SIM_AGG == "median" else float(np.mean(sims))
        shape_similarity.append(sim)

        prev_contours.append(cnt)
        prev_centroid = contour_centroid(cnt)

    out = np.column_stack([np.asarray(enclosed_pixels, dtype=np.float32),
                           np.asarray(shape_similarity, dtype=np.float32)])
    np.save(csv_save_path, out)
    print(f"Saved {video_name}: shape {out.shape} to {csv_save_path}")

# Model and paths
model = ultralytics.YOLO('/home/tarislada/Documents/Extra_python_projects/Natalie/best.pt')
video_path = '/home/tarislada/Documents/Extra_python_projects/Natalie/tst_pipeline/data/KHC2'  # Directory path
output_base_dir = '/home/tarislada/Documents/Extra_python_projects/Natalie/tst_pipeline/data/processed_videos'

os.makedirs(output_base_dir, exist_ok=True)

# Check if video_path is a directory or single file
if os.path.isdir(video_path):
    # Get all video files in the directory
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')
    video_files = [f for f in os.listdir(video_path) if f.endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {video_path}")
    else:
        print(f"Found {len(video_files)} video(s) to process")
        for video_file in video_files:
            full_video_path = os.path.join(video_path, video_file)
            process_video(full_video_path, model, output_base_dir)
else:
    # Process single video file
    process_video(video_path, model, output_base_dir)

print("\nAll processing finished!")
