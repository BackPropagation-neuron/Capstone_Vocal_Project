import os
from dotenv import load_dotenv

from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager
from Feedback_Generator.semantic_compressor import VocalSemanticCompressor
from Feedback_Generator.feedback_generator import VocalCoachLLM

def process_vocal_session(audio_file_path, user_id="test_user_001"):
    print("==================================================")
    print("[Step 1] 보컬 데이터 추출을 시작합니다...")
    print("==================================================")
    extractor = VocalFeatureExtractor(audio_file_path)
    features = extractor.process_all()
    print("-> 오디오 추출 및 연산 완료!\n")

    print("==================================================")
    print("[Step 2] 데이터베이스 및 S3(로컬) 저장을 진행합니다...")
    print("==================================================")
    db_manager = DBManager()
    session_id = db_manager.save_vocal_data(features, user_id)
    print("\n")

    print("==================================================")
    print("[Step 3] 데이터를 의미론적 텍스트로 압축합니다 (Semantic Compression)...")
    print("==================================================")
    compressor = VocalSemanticCompressor()
    semantic_text = compressor.compress_to_semantic_text(features)
    print("-> 텍스트 진단서 생성 완료!\n")

    print("==================================================")
    print("[Step 4] 보컬 트레이너(LLM) 피드백 생성을 시작합니다...")
    print("==================================================")
    # 환경변수에서 API 키를 가져옵니다.
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    coach = VocalCoachLLM(api_key=api_key, prompt_file_path='/home/ysm/Vocal_Project/Feedback_Generator/analysis_feedback_prompt.txt')
    feedback = coach.generate_feedback(semantic_text)
    print("\n[최종 생성된 코칭 피드백]")
    print(feedback)
    print("\n")

    print("==================================================")
    print("[Step 5] 생성된 피드백을 데이터베이스에 업데이트합니다...")
    print("==================================================")
    db_manager.update_feedback(session_id, feedback)
    print("\n모든 보컬 코칭 파이프라인이 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    # 테스트할 오디오 파일 경로를 지정합니다.
    test_audio_path = '/home/ysm/Vocal_Project/Vocal_Data/prompt.wav'
    
    if os.path.exists(test_audio_path):
        process_vocal_session(test_audio_path)
    else:
        print(f"파일을 찾을 수 없습니다: {test_audio_path}")
        print("먼저 분석할 오디오 파일을 준비해 주세요.")