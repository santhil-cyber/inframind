from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite Database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./crack_detection.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Model
class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    file_type = Column(String) # 'image' or 'video'
    crack_count = Column(Integer)
    max_severity = Column(String) # 'minor', 'moderate', 'severe'
    avg_confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    result_path = Column(String) # Path to annotated file

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables
Base.metadata.create_all(bind=engine)