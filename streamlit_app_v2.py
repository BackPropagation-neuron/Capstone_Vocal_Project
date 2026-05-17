import os
import streamlit as st
import tempfile
import librosa
import soundfile as sf

# 우리가 만든 V2 모듈들 임포트
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager
from Feedback_Generator_V2.semantic_compressor_v2 import VocalSemanticCompressorV2
from Feedback_Generator_V2.feedback_generator_v2 import VocalCoachChatSession

# ---------------------------------------------------------
# 유틸리티 함수
# ---------------------------------------------------------
def trim_silence_for_gemini(input_audio_path, output_audio_path="trimmed_temp.wav"):
    y, sr = librosa.load(input_audio_path, sr=None)
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    sf.write(output_audio_path, y_trimmed, sr)
    return output_audio_path

# ---------------------------------------------------------
# Streamlit 상태 초기화 (세션 유지)
# ---------------------------------------------------------
# 대화 내역 저장용
if "messages" not in st.session_state:
    st.session_state.messages = []
# 제미나이 채팅 객체 저장용 (상태 유지)
if "coach_session" not in st.session_state:
    st.session_state.coach_session = None
# DB 세션 ID 저장용
if "db_session_id" not in st.session_state:
    st.session_state.db_session_id = None

# ---------------------------------------------------------
# UI 레이아웃
# ---------------------------------------------------------
st.set_page_config(page_title="AI Vocal Coach V2", page_icon="🎤", layout="wide")
st.title("🎤 AI 보컬 마스터 코치 (V2)")
st.markdown("당신의 목소리를 분석하고, 전공 서적에 기반한 전문적인 피드백을 대화형으로 제공합니다.")

# 사이드바: 파일 업로드 및 분석 요청
with st.sidebar:
    st.header("보컬 데이터 입력")
    uploaded_file = st.file_uploader("보컬 오디오 파일 업로드 (.wav)", type=["wav"])
    
    st.subheader("💡 특별 요청사항 (Focus Point)")
    user_request = st.text_area("코치에게 특별히 묻고 싶은 점을 적어주세요.", 
                                placeholder="예: 고음 낼 때 목이 너무 조여요. 호흡과 턱 위치 위주로 봐주세요.")
    
    analyze_button = st.button("🚀 분석 및 코칭 시작", use_container_width=True)

# ---------------------------------------------------------
# 분석 파이프라인 실행
# ---------------------------------------------------------
if analyze_button and uploaded_file is not None:
    # 1. 파일 임시 저장 (Librosa 처리를 위해)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_audio_path = tmp_file.name

    with st.spinner("전문 보컬 데이터를 추출 중입니다... (약 10~20초 소요)"):
        # 2. 특징 추출
        extractor = VocalFeatureExtractor(tmp_audio_path)
        features = extractor.process_all()
        
        # [1차 방어선] 완전한 무음이나 악기 소리 차단
        vocal_confidence = features.get('voiced_probs_mean', 0.0)
        if vocal_confidence < 0.20:
            st.error("🚨 성대 진동이 거의 감지되지 않았습니다. 악기 소리이거나 무음으로 추정됩니다. 다시 녹음해 주세요!")
            os.remove(tmp_audio_path)
            st.stop()
            
        # 3. DB 저장
        db_manager = DBManager()
        st.session_state.db_session_id = db_manager.save_vocal_data(features, user_id="web_user_01")
        
        # 4. Semantic Compression
        compressor = VocalSemanticCompressorV2()
        semantic_text = compressor.compress_to_semantic_text(features)
        
        # 5. 제미나이 전송용 오디오 트리밍
        trimmed_audio_path = trim_silence_for_gemini(tmp_audio_path)

    with st.spinner("🤖 AI 코치가 소리를 듣고 리포트를 작성 중입니다..."):
        # 6. 채팅 세션 초기화 및 최초 리포트 생성
        st.session_state.coach_session = VocalCoachChatSession(db_path="./vocal_rag_db")
        initial_feedback = st.session_state.coach_session.start_initial_analysis(
            semantic_text=semantic_text,
            audio_file_path=trimmed_audio_path,
            user_request=user_request
        )
        
        # 7. DB에 피드백 업데이트
        db_manager.update_feedback(st.session_state.db_session_id, initial_feedback)
        
        # 8. 대화 내역에 추가 (초기화)
        st.session_state.messages = [{"role": "assistant", "content": initial_feedback}]
        
        # 임시 파일 정리
        os.remove(tmp_audio_path)
        if os.path.exists(trimmed_audio_path):
            os.remove(trimmed_audio_path)

elif analyze_button and uploaded_file is None:
    st.warning("⚠️ 먼저 오디오 파일을 업로드해 주세요.")

# ---------------------------------------------------------
# 채팅 인터페이스 (Chat UI)
# ---------------------------------------------------------
st.divider()

# 기존 대화 내역 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 꼬리 질문 입력창
if prompt := st.chat_input("추가로 궁금한 점을 코치에게 물어보세요!"):
    if st.session_state.coach_session is None:
        st.warning("⚠️ 먼저 오디오를 업로드하고 분석을 시작해 주세요.")
    else:
        # 사용자 메시지 화면에 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # 제미나이 후속 질문 호출 (Context 유지됨!)
        with st.chat_message("assistant"):
            with st.spinner("코치가 답변을 고민 중입니다..."):
                response = st.session_state.coach_session.send_followup_message(prompt)
                st.markdown(response)
        
        # 답변 저장
        st.session_state.messages.append({"role": "assistant", "content": response})