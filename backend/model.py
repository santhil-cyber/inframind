import cv2
import numpy as np
import uuid
import os
import tempfile
from collections import Counter

TEMP_DIR = os.path.join(tempfile.gettempdir(), "crack_detection_media")
os.makedirs(TEMP_DIR, exist_ok=True)

# --- Try loading specialized YOLO crack model (optional enhancer) ---
detection_model = None
HAS_CRACK_MODEL = False
try:
    from ultralytics import YOLO
    model_path = os.path.join(os.path.dirname(__file__), "crack.pt")
    if os.path.exists(model_path):
        detection_model = YOLO(model_path)
        HAS_CRACK_MODEL = True
        print("✅ Loaded specialized YOLOv8 crack model (will combine with CV pipeline)")
    else:
        print("ℹ️  'crack.pt' not found — using advanced CV-based crack detection pipeline")
except Exception as e:
    print(f"ℹ️  YOLO not available ({e}) — using CV-based crack detection pipeline")

print("✅ Advanced CV Crack + Corrosion + Spalling Detection Engine loaded")

# ============================================================
# RISK ANALYSIS
# ============================================================

def analyze_risk(defect_type, area_ratio, confidence):
    """
    Determine risk level and corrective action based on defect characteristics.
    """
    risk = "Low"
    action = "Monitor condition regularly."
    
    if defect_type == "Corrosion":
        if area_ratio > 5.0 or confidence > 0.85:
            risk = "High"
            action = "Immediate material replacement required. **Implement cathodic protection system.**"
        elif area_ratio > 1.0:
            risk = "Medium"
            action = "Remove corroded material, treat with rust inhibitor, and reinforce."
        else:
            action = "Clean surface and apply protective coating to prevent spread."
            
    elif defect_type == "Crack":
        if area_ratio > 2.0 or confidence > 0.90:
            risk = "High"
            action = "Structural integrity compromised. **Evacuate area and shore up immediately.**"
        elif area_ratio > 0.3 or confidence > 0.75:
            risk = "Medium"
            action = "Inject epoxy resin and install monitoring gauges."
        else:
            action = "Seal with flexible sealant to prevent water ingress."
            
    elif defect_type == "Spalling":
        if area_ratio > 3.0:
            risk = "High"
            action = "Area unsafe. Remove loose material and perform full depth patch repair."
        else:
            risk = "Medium"
            action = "Patch with polymer-modified repair mortar."
            
    return risk, action

# ============================================================
# ADVANCED CV-BASED CRACK DETECTION
# ============================================================

def _enhance_for_cracks(img):
    """
    Multi-stage image enhancement to make cracks more visible.
    Uses CLAHE, bilateral filtering, and sharpening.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Dramatically improves visibility of hairline cracks in low-contrast regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Step 2: Bilateral filter — smooths flat surfaces while keeping crack edges sharp
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Step 3: Unsharp masking for edge sharpening
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    
    return sharpened

def _detect_crack_edges(enhanced_gray, img_shape):
    """
    Multi-method edge detection optimized for crack patterns.
    Combines Canny, adaptive thresholding, and Laplacian for robustness.
    """
    height, width = img_shape[:2]
    
    # Method 1: Adaptive Gaussian Thresholding
    # Excellent for detecting dark cracks against varying background brightness
    adaptive = cv2.adaptiveThreshold(
        enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    
    # Method 2: Multi-scale Canny edge detection
    # Use Otsu's method to auto-determine thresholds
    v = np.median(enhanced_gray)
    lower = int(max(0, 0.5 * v))
    upper = int(min(255, 1.2 * v))
    canny = cv2.Canny(enhanced_gray, lower, upper)
    
    # Method 3: Laplacian of Gaussian (LoG) for fine cracks
    log = cv2.Laplacian(enhanced_gray, cv2.CV_64F, ksize=5)
    log = np.uint8(np.absolute(log))
    _, log_thresh = cv2.threshold(log, 30, 255, cv2.THRESH_BINARY)
    
    # Combine all three methods — a pixel is a crack candidate if detected by >=2 methods
    combined = np.zeros_like(enhanced_gray)
    combined[(adaptive > 0).astype(int) + (canny > 0).astype(int) + (log_thresh > 0).astype(int) >= 2] = 255
    
    return combined

def _filter_crack_contours(binary_mask, img_shape):
    """
    Filter contours using geometric properties specific to cracks:
    - Cracks are elongated (high aspect ratio)
    - Cracks have high arc-length-to-area ratio
    - Cracks are relatively thin
    """
    height, width = img_shape[:2]
    total_area = height * width
    
    # Morphological operations to connect nearby crack fragments and remove noise
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    connected = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_connect, iterations=2)
    kernel_connect_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_connect_v, iterations=2)
    
    # Remove tiny noise
    kernel_denoise = np.ones((3, 3), np.uint8)
    connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel_denoise, iterations=1)
    
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crack_contours = []
    min_contour_area = total_area * 0.0003  # min 0.03% of image area
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
            
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        
        # Get bounding box and rotated rect
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Crack-specific geometric filters
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        extent = area / (w * h + 1e-6)  # ratio of contour area to bounding box area
        compactness = (perimeter ** 2) / (4 * np.pi * area + 1e-6)  # circularity inverse
        
        # Cracks are elongated (aspect ratio > 2) and non-compact (compactness > 5)
        is_elongated = aspect_ratio > 1.8
        is_non_compact = compactness > 4.0
        is_thin = extent < 0.6  # cracks don't fill their bounding box
        
        # A contour is a crack if it meets elongation OR non-compactness criteria
        # (some cracks are branching/irregular, so we use OR logic)
        if is_elongated or (is_non_compact and is_thin):
            # Compute confidence based on how "crack-like" the shape is
            elongation_score = min(aspect_ratio / 10.0, 1.0)  # higher aspect = more confident
            compactness_score = min(compactness / 30.0, 1.0)
            size_score = min(area / (total_area * 0.01), 1.0)  # larger = more confident
            
            confidence = 0.55 + 0.15 * elongation_score + 0.15 * compactness_score + 0.15 * size_score
            confidence = min(confidence, 0.98)
            
            crack_contours.append({
                "contour": cnt,
                "bbox": (x, y, w, h),
                "area": area,
                "confidence": confidence,
                "aspect_ratio": aspect_ratio,
                "compactness": compactness
            })
    
    return crack_contours


def detect_cracks_cv(img, mask_output):
    """
    Advanced CV-based crack detection using multi-stage pipeline:
    1. Image enhancement (CLAHE + bilateral + sharpening)
    2. Multi-method edge detection (Adaptive + Canny + LoG)
    3. Geometric contour filtering (elongation, compactness, thinness)
    4. Confidence scoring based on crack-like properties
    """
    height, width = img.shape[:2]
    total_area = height * width
    
    # Stage 1: Enhance image for crack visibility
    enhanced = _enhance_for_cracks(img)
    
    # Stage 2: Multi-method edge detection
    crack_mask = _detect_crack_edges(enhanced, img.shape)
    
    # Stage 3: Filter contours by crack geometry
    crack_contours = _filter_crack_contours(crack_mask, img.shape)
    
    # Stage 4: Build detections and draw on output
    detections = []
    for cc in crack_contours:
        x, y, w, h = cc["bbox"]
        conf = cc["confidence"]
        area_ratio = (cc["area"] / total_area) * 100
        
        # Draw the actual contour shape (more accurate than just a box)
        cv2.drawContours(mask_output, [cc["contour"]], -1, (0, 0, 255), 2)
        # Draw bounding box
        cv2.rectangle(mask_output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(mask_output, f"Crack {conf:.0%}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        risk, action = analyze_risk("Crack", area_ratio, conf)
        
        detections.append({
            "type": "Crack",
            "confidence": round(conf, 2),
            "box": [x, y, x + w, y + h],
            "risk_level": risk,
            "corrective_action": action,
            "area_ratio": area_ratio
        })
    
    return detections


def detect_cracks_yolo(img, mask_output):
    """
    Detect cracks using specialized YOLOv8 Model (only if crack.pt exists).
    """
    if detection_model is None:
        return []

    results = detection_model(img, verbose=False)
    
    detections = []
    height, width = img.shape[:2]
    total_area = height * width

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            conf = float(box.conf[0])
            
            if conf < 0.20: continue

            label = "Crack" 
            
            w_box = x2 - x1
            h_box = y2 - y1
            area = w_box * h_box
            area_ratio = (area / total_area) * 100

            cv2.rectangle(mask_output, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(mask_output, f"{label} {conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            risk, action = analyze_risk("Crack", area_ratio, conf)
            
            detections.append({
                "type": label,
                "confidence": round(conf, 2),
                "box": [x1, y1, x2, y2],
                "risk_level": risk,
                "corrective_action": action,
                "area_ratio": area_ratio
            })
            
    return detections


def detect_cracks(img, mask_output):
    """
    Combined crack detection: CV pipeline always runs.
    If crack.pt YOLO model exists, its detections are merged and deduplicated.
    """
    # Always use the CV pipeline — it works on any concrete/asphalt image
    cv_detections = detect_cracks_cv(img, mask_output)
    
    # If we have the specialized YOLO model, merge its results too
    if HAS_CRACK_MODEL:
        yolo_detections = detect_cracks_yolo(img, mask_output)
        cv_detections = _merge_detections(cv_detections, yolo_detections)
    
    return cv_detections


def _merge_detections(cv_dets, yolo_dets):
    """
    Merge CV and YOLO detections, removing duplicates by IoU overlap.
    YOLO detections take priority when overlapping.
    """
    if not yolo_dets:
        return cv_dets
    if not cv_dets:
        return yolo_dets
    
    merged = list(yolo_dets)  # YOLO results are primary
    
    for cv_det in cv_dets:
        overlaps = False
        cv_box = cv_det["box"]
        for yolo_det in yolo_dets:
            yolo_box = yolo_det["box"]
            iou = _compute_iou(cv_box, yolo_box)
            if iou > 0.3:  # significant overlap
                overlaps = True
                break
        if not overlaps:
            merged.append(cv_det)
    
    return merged


def _compute_iou(box1, box2):
    """Compute Intersection over Union between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)


# ============================================================
# CORROSION DETECTION (Enhanced)
# ============================================================

def detect_corrosion(img, mask_output):
    """
    Detect corrosion (rust) using multi-range HSV color thresholding.
    Enhanced with multiple rust color ranges for better coverage.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Multiple rust/corrosion color ranges for broader detection
    rust_ranges = [
        (np.array([0, 80, 30]),   np.array([15, 255, 255])),   # Red-orange rust
        (np.array([15, 80, 20]),  np.array([25, 255, 255])),   # Orange-brown rust
        (np.array([160, 80, 30]), np.array([180, 255, 255])),  # Deep red rust (wraps around)
    ]
    
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in rust_ranges:
        mask |= cv2.inRange(hsv, lower, upper)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = img.shape[:2]
    total_area = height * width
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:  # Lowered from 500 for better sensitivity
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw contour and bounding box
            cv2.drawContours(mask_output, [cnt], -1, (0, 165, 255), 2)
            cv2.rectangle(mask_output, (x, y), (x + w, y + h), (0, 165, 255), 2)
            
            area_ratio = (area / total_area) * 100
            conf = 0.80 + min((area / 5000) * 0.1, 0.15) 
            
            cv2.putText(mask_output, f"Corrosion {conf:.0%}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            risk, action = analyze_risk("Corrosion", area_ratio, conf)
            
            detections.append({
                "type": "Corrosion",
                "confidence": round(conf, 2),
                "box": [x, y, x + w, y + h],
                "risk_level": risk,
                "corrective_action": action,
                "area_ratio": area_ratio
            })
            
    return detections


# ============================================================
# SPALLING DETECTION (New!)
# ============================================================

def detect_spalling(img, mask_output):
    """
    Detect spalling (surface flaking/chipping) using texture analysis.
    Looks for rough, irregular texture regions that differ from smooth concrete.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    total_area = height * width
    
    # Compute local texture roughness using standard deviation filter
    kernel_size = 15
    local_mean = cv2.blur(gray.astype(np.float64), (kernel_size, kernel_size))
    local_sq_mean = cv2.blur((gray.astype(np.float64)) ** 2, (kernel_size, kernel_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))
    
    # High texture variance = rough / damaged surface
    std_norm = (local_std / (local_std.max() + 1e-6) * 255).astype(np.uint8)
    _, rough_mask = cv2.threshold(std_norm, 120, 255, cv2.THRESH_BINARY)
    
    # Combine with brightness analysis — spalling areas are often lighter (exposed aggregate)
    _, bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    combined = cv2.bitwise_and(rough_mask, bright_mask)
    
    # Morphological cleanup
    kernel = np.ones((11, 11), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    min_area = total_area * 0.005  # spalling regions tend to be larger
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
        
        # Spalling is typically blob-like (not elongated like cracks)
        if aspect_ratio < 5.0:
            area_ratio = (area / total_area) * 100
            conf = 0.60 + min(area_ratio / 10.0, 0.3)
            
            cv2.drawContours(mask_output, [cnt], -1, (255, 0, 255), 2)
            cv2.rectangle(mask_output, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(mask_output, f"Spalling {conf:.0%}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            risk, action = analyze_risk("Spalling", area_ratio, conf)
            
            detections.append({
                "type": "Spalling",
                "confidence": round(conf, 2),
                "box": [x, y, x + w, y + h],
                "risk_level": risk,
                "corrective_action": action,
                "area_ratio": area_ratio
            })
    
    return detections


# ============================================================
# MATERIAL DETECTION
# ============================================================

def detect_material(img):
    """
    Estimate material (Concrete vs Asphalt/Pavement) based on color statistics.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    
    avg_l = np.mean(l)
    
    if avg_l < 80:
        return "Asphalt/Pavement"
    else:
        return "Concrete"


# ============================================================
# HEATMAP GENERATION
# ============================================================

def generate_heatmap(img, detections):
    """
    Generate a heatmap overlay based on detection density.
    """
    height, width = img.shape[:2]
    heatmap_mask = np.zeros((height, width), dtype=np.float32)
    
    for d in detections:
        x1, y1, x2, y2 = d['box']
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        
        weight = 1.0
        if d['risk_level'] == 'High': weight = 3.0
        elif d['risk_level'] == 'Medium': weight = 2.0
        
        radius = int(max(x2-x1, y2-y1) * 1.5)
        cv2.circle(heatmap_mask, (cx, cy), radius, (weight), -1)
        
    heatmap_mask = cv2.GaussianBlur(heatmap_mask, (121, 121), 0)
    
    if np.max(heatmap_mask) > 0:
        heatmap_mask = (heatmap_mask / np.max(heatmap_mask)) * 255
    
    heatmap_mask = np.uint8(heatmap_mask)
    heatmap_color = cv2.applyColorMap(heatmap_mask, cv2.COLORMAP_INFERNO)
    
    overlay = cv2.addWeighted(img, 0.7, heatmap_color, 0.5, 0)
    return overlay


# ============================================================
# IMAGE PROCESSING PIPELINE
# ============================================================

def process_image(file_bytes: bytes, filename: str):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")

    output_img = img.copy()
    
    material = detect_material(img)
    
    # Run all three detectors
    corrosion_results = detect_corrosion(img, output_img)
    crack_results = detect_cracks(img, output_img)
    spalling_results = detect_spalling(img, output_img)
    
    all_detections = corrosion_results + crack_results + spalling_results
    
    # Generate heatmap
    heatmap_img = generate_heatmap(img, all_detections)
    
    # Save outputs
    base_id = uuid.uuid4().hex[:8]
    output_filename = f"processed_{base_id}_{filename}"
    heatmap_filename = f"heatmap_{base_id}_{filename}"
    
    output_path = os.path.join(TEMP_DIR, output_filename)
    heatmap_path = os.path.join(TEMP_DIR, heatmap_filename)
    
    cv2.imwrite(output_path, output_img)
    cv2.imwrite(heatmap_path, heatmap_img)
    
    max_severity = "minor"
    risks = [d['risk_level'] for d in all_detections]
    if "High" in risks:
        max_severity = "severe"
    elif "Medium" in risks:
        max_severity = "moderate"
    elif not all_detections:
        max_severity = "none"

    return output_path, heatmap_path, all_detections, max_severity, material


# ============================================================
# VIDEO PROCESSING PIPELINE
# ============================================================

def process_video(file_bytes: bytes, filename: str):
    base_id = uuid.uuid4().hex[:8]
    input_filename = f"input_{base_id}_{filename}"
    output_filename = f"processed_{base_id}_{filename}"
    
    input_path = os.path.join(TEMP_DIR, input_filename)
    output_path = os.path.join(TEMP_DIR, output_filename)
    
    with open(input_path, "wb") as f:
        f.write(file_bytes)
        
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frames_processed = 0
    
    SKIP_FRAMES = 4
    current_detections_c = []
    current_detections_k = []
    current_detections_s = []
    
    print(f"Starting video processing: {width}x{height} @ {fps}fps, {total_frames} frames. Processing every {SKIP_FRAMES+1} frames.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = frame.copy()
        
        if frames_processed % (SKIP_FRAMES + 1) == 0:
            current_detections_c = detect_corrosion(frame, output_frame)
            current_detections_k = detect_cracks(frame, output_frame)
            current_detections_s = detect_spalling(frame, output_frame)
            
            all_detections.extend(current_detections_c)
            all_detections.extend(current_detections_k)
            all_detections.extend(current_detections_s)
        else:
            # Replay previous detections on skipped frames
            for d in current_detections_c + current_detections_k + current_detections_s:
                x1, y1, x2, y2 = d['box']
                if d['type'] == 'Crack':
                    color = (0, 0, 255)
                elif d['type'] == 'Corrosion':
                    color = (0, 165, 255)
                else:
                    color = (255, 0, 255)
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        out.write(output_frame)
        frames_processed += 1
        
        if frames_processed % 30 == 0:
            print(f"Processed {frames_processed}/{total_frames}")

    cap.release()
    out.release()
    
    try:
        os.remove(input_path)
    except:
        pass

    max_severity = "minor"
    risks = [d['risk_level'] for d in all_detections]
    if "High" in risks:
        max_severity = "severe"
    elif "Medium" in risks:
        max_severity = "moderate"
    elif not all_detections:
        max_severity = "none"

    if len(all_detections) > 100:
        high_risk = [d for d in all_detections if d['risk_level'] == 'High']
        other_risk = [d for d in all_detections if d['risk_level'] != 'High']
        all_detections = high_risk + other_risk[:50]
        
    return output_path, all_detections, max_severity
