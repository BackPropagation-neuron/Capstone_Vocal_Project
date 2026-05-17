import os
import time
import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

# [V1 기존 모듈]
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager

# [V2 신규 모듈]
from Feedback_Generator_V2.semantic_compressor_v2 import VocalSemanticCompressorV2
from Feedback_Generator_V2.feedback_generator_v2 import VocalCoachChatSession

def trim_silence_for_gemini(input_audio_path, output_audio_path="trimmed_temp.wav"):
    """
    [V2 신규 기능] 오디오의 앞뒤 무음을 제거하여 Gemini가 핵심 발성에만 집중하도록 돕습니다.
    """
    print("오디오 전처리: 앞뒤 무음을 제거 중입니다...")
    y, sr = librosa.load(input_audio_path, sr=None)
    # top_db=30: 최대 볼륨 대비 -30dB 이하의 소리는 무음으로 간주하고 자름
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    
    sf.write(output_audio_path, y_trimmed, sr)
    print(f"무음 제거 완료. (임시 파일: {output_audio_path})")
    return output_audio_path

def process_vocal_session_v2(audio_file_path, user_id="test_user_v2", user_request=None):
    print("\n" + "="*50)
    print("[Step 1] 보컬 데이터 정밀 추출을 시작합니다...")
    print("="*50)
    extractor = VocalFeatureExtractor(audio_file_path)
    features = extractor.process_all()
    
    vocal_confidence = features.get('voiced_probs_mean', 0.0)
    print(f"보컬 신뢰도(Vocal Confidence): {vocal_confidence:.2f}")
    
    if vocal_confidence < 0.20:
        print("반려됨: 보컬이 명확하게 감지되지 않았습니다. 분석을 중단합니다.")
        return "보컬이 명확하게 감지되지 않았습니다. 악기 소리나 주변 잡음이 너무 큰 것 같아요. 목소리가 잘 들리도록 다시 녹음해 주세요!"
    
    print("-> 오디오 추출 및 연산 완료!")

    print("\n" + "="*50)
    print("[Step 2] 데이터베이스 및 S3(로컬) 저장을 진행합니다...")
    print("="*50)
    db_manager = DBManager()
    session_id = db_manager.save_vocal_data(features, user_id)

    print("\n" + "="*50)
    print("[Step 3] 데이터를 V2 규격의 의미론적 텍스트로 압축합니다...")
    print("="*50)
    compressor = VocalSemanticCompressorV2()
    semantic_text = compressor.compress_to_semantic_text(features)
    print("-> 텍스트 진단 리포트 기초 데이터 생성 완료!")

    print("\n" + "="*50)
    print("[Step 4] Gemini 멀티모달 전송을 위한 오디오 전처리...")
    print("="*50)
    # 제미나이에게 넘겨줄 알맹이만 남긴 임시 오디오 파일 생성
    trimmed_audio_path = trim_silence_for_gemini(audio_file_path)

    print("\n" + "="*50)
    print("[Step 5] 대화형 보컬 코치(LLM) 세션을 시작합니다...")
    print("="*50)
    # 1. 채팅 세션 객체 생성 (RAG DB 로드 포함)
    coach_session = VocalCoachChatSession(db_path="./vocal_rag_db") # DB 경로 확인 필요
    
    # 2. 최초 분석 리포트 생성 (오디오 + 수치데이터 + 사용자 요청사항 + RAG 지식)
    initial_feedback = coach_session.start_initial_analysis(
        semantic_text=semantic_text,
        audio_file_path=trimmed_audio_path,
        user_request=user_request
    )
    
    print("\n" + "*"*50)
    print("[최초 생성된 V2 멀티모달 코칭 리포트]")
    print("*"*50)
    print(initial_feedback)
    print("*"*50)

    print("\n" + "="*50)
    print("[Step 6] 생성된 메인 피드백을 DB에 업데이트합니다...")
    print("="*50)
    db_manager.update_feedback(session_id, initial_feedback)

    print("\n" + "="*50)
    print("[Step 7] 대화 세션(Chat State) 유지 테스트를 진행합니다...")
    print("="*50)
    # 제미나이가 과거 오디오와 데이터를 잘 기억하는지 꼬리 질문을 던져봅니다.
    follow_up_question = "제가 방금 부른 소리에서, 연습할 때 턱을 얼마나 더 벌려야 할까요? 팁을 주세요."
    print(f"사용자 질문: {follow_up_question}")
    
    follow_up_answer = coach_session.send_followup_message(follow_up_question)
    print(f"\n코치 답변:\n{follow_up_answer}")

    # 테스트가 끝난 후 임시 파일 삭제
    if os.path.exists(trimmed_audio_path):
        os.remove(trimmed_audio_path)
        print("\n임시 오디오 파일이 삭제되었습니다.")

    print("\n모든 V2 파이프라인(멀티모달 + RAG + Chat) 테스트가 성공적으로 종료되었습니다!")

if __name__ == "__main__":
    # 테스트할 원본 오디오 파일 경로
    test_audio_path = '/home/ysm/Vocal_Project/Vocal_Data/prompt.wav'
    
    # UI에서 입력받았다고 가정한 사용자의 특별 요청사항
    sample_user_request = "최근에 고음을 낼 때 목이 너무 조여서 답답해요. 호흡과 턱의 위치를 집중해서 분석해 주세요."
    
    # 주의: 터미널에서 실행하는 위치(Current Working Directory)에 
    # `vocal_rag_db` 폴더가 잘 보이는지 확인 후 실행하세요.
    
    if os.path.exists(test_audio_path):
        process_vocal_session_v2(
            audio_file_path=test_audio_path, 
            user_request=sample_user_request
        )
    else:
        print(f"파일을 찾을 수 없습니다: {test_audio_path}")