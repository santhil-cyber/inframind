import cv2
import numpy as np
import uuid
import os
import tempfile
from collections import Counter
from ultralytics import YOLO

TEMP_DIR = os.path.join(tempfile.gettempdir(), "crack_detection_media")
os.makedirs(TEMP_DIR, exist_ok=True)

print("Loading Crack Detection Model...")
try:
    model_path = os.path.join(os.path.dirname(__file__), "crack.pt")
    if os.path.exists(model_path):
        detection_model = YOLO(model_path)
        print("✅ Loaded specialized YOLOv8 crack model")
    else:
        print("⚠️ 'crack.pt' not found, falling back to standard yolov8x.pt")
        detection_model = YOLO("yolov8x.pt")
except Exception as e:
    print(f"Error loading model: {e}")
    detection_model = None

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
        if area_ratio > 2.0:
            risk = "High"
            action = "Structural integrity compromised. **Evacuate area and shore up immediately.**"
        elif area_ratio > 0.5:
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

def detect_corrosion(img, mask_output):
    """
    Detect corrosion (rust) using HSV color thresholding.
    Returns list of detections.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
   
    lower_rust = np.array([10, 100, 20])
    upper_rust = np.array([25, 255, 255])
    
    mask = cv2.inRange(hsv, lower_rust, upper_rust)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = img.shape[:2]
    total_area = height * width
    detections = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500: 
            x, y, w, h = cv2.boundingRect(cnt)
            
            cv2.rectangle(mask_output, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(mask_output, "Corrosion", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
            area_ratio = (area / total_area) * 100
            conf = 0.80 + min((area/5000) * 0.1, 0.15) 
            
            risk, action = analyze_risk("Corrosion", area_ratio, conf)
            
            detections.append({
                "type": "Corrosion",
                "confidence": round(conf, 2),
                "box": [x, y, x+w, y+h],
                "risk_level": risk,
                "corrective_action": action,
                "area_ratio": area_ratio
            })
            
    return detections

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

def detect_cracks(img, mask_output):
    """
    Detect cracks using YOLOv8 Model.
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
            
            if conf < 0.25: continue

            cls = int(box.cls[0])
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

def process_image(file_bytes: bytes, filename: str):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Could not decode image")

    output_img = img.copy()
    
    material = detect_material(img)
    
    corrosion_results = detect_corrosion(img, output_img)
    crack_results = detect_cracks(img, output_img)
    
    all_detections = corrosion_results + crack_results
    
    heatmap_img = generate_heatmap(img, all_detections)
    
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
    
    print(f"Starting video processing: {width}x{height} @ {fps}fps, {total_frames} frames. Optimization: Processing every {SKIP_FRAMES+1} frames.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = frame.copy()
        
        if frames_processed % (SKIP_FRAMES + 1) == 0:
            current_detections_c = detect_corrosion(frame, output_frame) # This draws on output_frame
            current_detections_k = detect_cracks(frame, output_frame)    # This draws on output_frame (redrawing over same frame)
            
            all_detections.extend(current_detections_c)
            all_detections.extend(current_detections_k)
        else:
           
            for d in current_detections_c + current_detections_k:
                x1, y1, x2, y2 = d['box']
                color = (0, 0, 255) if d['type'] == 'Crack' else (0, 165, 255) # Red or Orange
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
