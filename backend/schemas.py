from pydantic import BaseModel
from typing import List, Optional

class DefectDetail(BaseModel):
    id: int
    type: str  # "crack", "corrosion", "spalling", "other"
    confidence: float
    risk_level: str  # "High", "Medium", "Low"
    corrective_action: str
    box: List[int]  # [x1, y1, x2, y2]
    severity: str # Kept for backward compatibility

class AnalysisResponse(BaseModel):
    id: int
    filename: str
    cracks_detected: int # Total defects count
    max_severity: str
    avg_confidence: float
    avg_confidence: float
    file_type: str = "image" # "image" or "video"
    result_url: str
    heatmap_url: Optional[str] = None
    material_type: str = "Unknown"
    details: List[DefectDetail]
    
    # New statistics
    stats: dict = {} # e.g., {"critical": 1, "high": 2}

class HistoryItem(BaseModel):
    id: int
    filename: str
    date: str
    crack_count: int
    max_severity: str