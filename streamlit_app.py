import streamlit as st
import plotly.graph_objects as go
import os
import tempfile
from dotenv import load_dotenv

# 기존 모듈 임포트
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager
from Feedback_Generator.semantic_compressor import VocalSemanticCompressor
from Feedback_Generator.feedback_generator import VocalCoachLLM

# 환경 변수 로드
load_dotenv()

st.set_page_config(page_title="AI Vocal Coach Dashboard", layout="wide")

st.title("🎤 AI 보컬 코칭 분석 대시보드")
st.markdown("음성 파일을 업로드하면 AI가 발성 상태를 정밀 분석하고 맞춤형 피드백을 제공합니다.")

# 사이드바: 설정 및 업로드
with st.sidebar:
    st.header("파일 업로드")
    uploaded_file = st.file_uploader("WAV 파일을 선택하세요", type=["wav"])
    user_id = st.text_input("사용자 ID", value="test_user_001")
    analyze_btn = st.button("분석 시작", type="primary", disabled=not uploaded_file)

if analyze_btn and uploaded_file:
    # 1. 임시 파일 저장 (Extractor에서 경로를 필요로 하는 경우)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.status("보컬 데이터를 분석 중입니다...", expanded=True) as status:
            # Step 1: Feature Extraction
            st.write("🎵 오디오 특징 추출 중...")
            extractor = VocalFeatureExtractor(tmp_path)
            features = extractor.process_all()
            
            # Step 2: Semantic Compression
            st.write("🧠 데이터 의미 분석 중...")
            compressor = VocalSemanticCompressor()
            semantic_text = compressor.compress_to_semantic_text(features)
            
            # Step 3: LLM Feedback Generation
            st.write("🤖 AI 보컬 코치 피드백 생성 중...")
            coach = VocalCoachLLM(api_key=os.environ.get("GEMINI_API_KEY"), prompt_file_path='/home/ysm/Vocal_Project/Feedback_Generator/analysis_feedback_prompt.txt')
            feedback = coach.generate_feedback(semantic_text)
            
            # Step 4: DB Save (Optional)
            st.write("💾 분석 결과 저장 중...")
            db_manager = DBManager()
            session_id = db_manager.save_vocal_data(features, user_id)
            db_manager.update_feedback(session_id, feedback)
            
            status.update(label="분석 완료!", state="complete", expanded=False)

        # 결과 화면 구성
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🤖 AI 맞춤 피드백")
            st.info(feedback)
            
            st.subheader("📊 상세 진단 로그")
            st.code(semantic_text, language="markdown")

        with col2:
            st.subheader("📈 오디오 데이터 시각화")
            
            # Plotly를 이용한 F0(음정) 그래프
            f0_data = [x if x > 0 else None for x in features['f0_contour']]
            fig_f0 = go.Figure()
            fig_f0.add_trace(go.Scatter(y=f0_data, mode='lines', name='Pitch (Hz)'))
            fig_f0.update_layout(title="Pitch Contour (F0)", xaxis_title="Frame", yaxis_title="Frequency (Hz)")
            st.plotly_chart(fig_f0, use_container_width=True)
            
            # RMS(에너지) 그래프
            fig_rms = go.Figure()
            fig_rms.add_trace(go.Scatter(y=features['rms_contour'], mode='lines', name='Energy (RMS)'))
            fig_rms.update_layout(title="Volume Contour (RMS)", xaxis_title="Frame", yaxis_title="Amplitude")
            st.plotly_chart(fig_rms, use_container_width=True)

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("왼쪽 사이드바에서 WAV 파일을 업로드하고 '분석 시작' 버튼을 눌러주세요.")