import os
import json
import uuid
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. SQLAlchemy 기본 설정
DB_URL = "sqlite:///vocal_app.db"
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# 2. 데이터베이스 테이블 구조(Schema) 정의
class VocalSession(Base):
    __tablename__ = "vocal_sessions"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)              
    created_at = Column(DateTime, default=datetime.utcnow)
    
    contour_file_path = Column(String) 
    
    f0_mean = Column(Float)
    vibrato_rate_hz = Column(Float)
    jitter_percent = Column(Float)
    shimmer_percent = Column(Float)
    hnr_db = Column(Float)
    rms_mean = Column(Float)
    centroid_mean = Column(Float)
    
    mfcc_mean = Column(String) 
    ai_feedback = Column(String, default="")

Base.metadata.create_all(bind=engine)

class DBManager:
    def __init__(self):
        self.storage_dir = os.path.join(os.path.dirname(__file__), "local_s3_storage")
        os.makedirs(self.storage_dir, exist_ok=True)

    def _save_contours_to_local_s3(self, session_id, features):
        """거대한 시계열 데이터만 추출하여 JSON 파일로 저장합니다."""
        
        # [수정 핵심] float(x)를 통해 numpy 타입을 순수 파이썬 float으로 강제 변환합니다.
        contours_data = {
            "f0_contour": [float(x) for x in features.get("f0_contour", [])],
            "f1_contour": [float(x) for x in features.get("f1_contour", [])],
            "f2_contour": [float(x) for x in features.get("f2_contour", [])],
            "f3_contour": [float(x) for x in features.get("f3_contour", [])],
            "rms_contour": [float(x) for x in features.get("rms_contour", [])]
        }
        
        file_path = os.path.join(self.storage_dir, f"{session_id}_contours.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(contours_data, f)
            
        return file_path

    def save_vocal_data(self, features, user_id="dummy_user_123"):
        """전체 파이프라인: 시계열은 파일로, 요약은 DB로 저장합니다."""
        session_id = str(uuid.uuid4())
        
        contour_path = self._save_contours_to_local_s3(session_id, features)
        
        db_session = SessionLocal()
        try:
            # [수정 핵심] mfcc_mean 배열 내부의 값들도 float으로 안전하게 변환합니다.
            safe_mfcc = [float(x) for x in features.get("mfcc_mean", [])]
            
            new_record = VocalSession(
                id=session_id,
                user_id=user_id,
                contour_file_path=contour_path,
                f0_mean=float(features.get("f0_mean", 0.0)),
                vibrato_rate_hz=float(features.get("vibrato_rate_hz", 0.0)),
                jitter_percent=float(features.get("jitter_percent", 0.0)),
                shimmer_percent=float(features.get("shimmer_percent", 0.0)),
                hnr_db=float(features.get("hnr_db", 0.0)),
                rms_mean=float(features.get("rms_mean", 0.0)),
                centroid_mean=float(features.get("centroid_mean", 0.0)),
                mfcc_mean=json.dumps(safe_mfcc)
            )
            
            db_session.add(new_record)
            db_session.commit()
            print(f"DB 저장 완료! (Session ID: {session_id})")
            return session_id
            
        except Exception as e:
            db_session.rollback()
            print(f"DB 저장 중 오류 발생: {e}")
        finally:
            db_session.close()
    
    def update_feedback(self, session_id, feedback_text):
        """생성된 AI 피드백을 기존 DB 레코드에 업데이트합니다."""
        db_session = SessionLocal()
        try:
            # session_id로 해당 기록을 찾습니다.
            record = db_session.query(VocalSession).filter(VocalSession.id == session_id).first()
            if record:
                record.ai_feedback = feedback_text
                db_session.commit()
                print(f"DB 업데이트 완료: 피드백이 저장되었습니다. (Session ID: {session_id})")
            else:
                print(f"DB 오류: 해당 Session ID({session_id})를 찾을 수 없습니다.")
        except Exception as e:
            db_session.rollback()
            print(f"DB 업데이트 중 오류 발생: {e}")
        finally:
            db_session.close()