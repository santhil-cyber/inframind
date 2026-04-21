from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from typing import List
import os
import database
import model
from schemas import AnalysisResponse, HistoryItem

app = FastAPI(title="Inframind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/media", StaticFiles(directory=model.TEMP_DIR), name="media")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_file(
    file: UploadFile = File(...), 
    db: Session = Depends(database.get_db)
):
    file_bytes = await file.read()
    
    result_path = ""
    detections = []
    severity = "minor"
    
    try:
        if file.content_type.startswith("image"):
            result_path, heatmap_path, detections, severity, material = model.process_image(file_bytes, file.filename)
        elif file.content_type.startswith("video"):
            result_path, detections, severity = model.process_video(file_bytes, file.filename)
            heatmap_path = ""
            material = "Unknown"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    count = len(detections)
    if count > 0:
        avg_conf = sum(d['confidence'] for d in detections) / count
    else:
        avg_conf = 0.0
    
    severity_counts = {
        "Critical": len([d for d in detections if d.get('risk_level') == 'Critical']),
        "High": len([d for d in detections if d.get('risk_level') == 'High']),
        "Medium": len([d for d in detections if d.get('risk_level') == 'Medium']),
        "Low": len([d for d in detections if d.get('risk_level') == 'Low']),
    }

   
    db_record = database.AnalysisResult(
        filename=file.filename,
        file_type="image" if file.content_type.startswith("image") else "video",
        crack_count=count, 
        max_severity=severity,
        avg_confidence=round(avg_conf, 2),
        result_path=result_path
    )
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    
   
    formatted_details = []
    for i, d in enumerate(detections):
        formatted_details.append({
            "id": i,
            "type": d['type'],
            "confidence": d['confidence'],
            "risk_level": d['risk_level'],
            "corrective_action": d['corrective_action'],
            "box": d['box'],
            "severity": d['risk_level'].lower() # Mapping risk to severity for now
        })

    return AnalysisResponse(
        id=db_record.id,
        filename=file.filename,
        cracks_detected=count,
        max_severity=severity,
        avg_confidence=round(avg_conf, 2),
        file_type="image" if file.content_type.startswith("image") else "video",
        result_url=f"http://localhost:8000/media/{os.path.basename(result_path)}",
        heatmap_url=f"http://localhost:8000/media/{os.path.basename(heatmap_path)}" if heatmap_path else None,
        material_type=material,
        details=formatted_details,
        stats=severity_counts
    )

@app.get("/history", response_model=List[HistoryItem])
def get_history(db: Session = Depends(database.get_db)):
    records = db.query(database.AnalysisResult).order_by(database.AnalysisResult.timestamp.desc()).limit(20).all()
    return [
        HistoryItem(
            id=r.id,
            filename=r.filename,
            date=r.timestamp.strftime("%Y-%m-%d %H:%M"),
            crack_count=r.crack_count,
            max_severity=r.max_severity
        ) for r in records
    ]

