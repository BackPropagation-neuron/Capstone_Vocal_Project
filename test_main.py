# test_main.py
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager

def run_pipeline(audio_file_path, sr=22500):
    print("1. 오디오 피처 추출을 시작합니다...")
    extractor = VocalFeatureExtractor(audio_file_path, sr=sr)
    features = extractor.process_all()
    
    print("\n2. 추출된 데이터를 데이터베이스와 스토리지에 분리 저장합니다...")
    db_manager = DBManager()
    session_id = db_manager.save_vocal_data(features, user_id="test_user_001")
    
    print(f"\n파이프라인 실행 완료! 저장된 세션 ID: {session_id}")

if __name__ == "__main__":
    # 준비해둔 테스트용 wav 파일 경로를 입력하세요
    # (앞서 만든 generate_dummy_vocal 함수로 만든 파일을 쓰시면 됩니다)
    test_sr = 24000
    test_audio_path = '/home/ysm/Vocal_Project/Vocal_Data/prompt.wav'
    run_pipeline(test_audio_path)