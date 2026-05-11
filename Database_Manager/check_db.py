# check_db.py
from database_manager import SessionLocal, VocalSession

def check_database():
    # 1. DB 장부를 엽니다.
    db_session = SessionLocal()
    
    # 2. VocalSession 테이블에 있는 모든 기록을 가져옵니다.
    records = db_session.query(VocalSession).all()
    
    print("========================================")
    print(f"현재 DB에 저장된 총 세션 수: {len(records)}개")
    print("========================================\n")
    
    # 3. 하나씩 꺼내서 보기 좋게 출력합니다.
    for record in records:
        print(f"세션 ID: {record.id}")
        print(f"유저 ID: {record.user_id}")
        print(f"분석 일시: {record.created_at}")
        print(f"평균 피치 (F0): {record.f0_mean:.2f} Hz")
        print(f"평균 HNR: {record.hnr_db:.2f} dB")
        print(f"JSON 파일 위치: {record.contour_file_path}")
        print(f"AI 피드백 내용: '{record.ai_feedback}'")
        print("-" * 40)
        
    # 4. 장부를 닫습니다.
    db_session.close()

if __name__ == "__main__":
    check_database()