import streamlit as st
import plotly.graph_objects as go
import os
import tempfile
import base64
import numpy as np
from PIL import Image
from dotenv import load_dotenv

# 기존 모듈 임포트
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager
from Feedback_Generator.semantic_compressor import VocalSemanticCompressor
from Feedback_Generator.feedback_generator import VocalCoachLLM

# 환경 변수 로드
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

# --- 상대 경로 설정 (팀원 공유 최적화) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "vocal_coach.png")

# 로컬 이미지를 HTML에서 사용하기 위한 변환 함수
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image(IMAGE_PATH)

# --- 스타일 UI 정의 ---
ST_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;500;700&display=swap');
    .stApp { background-color: #121212; color: #ffffff; font-family: 'Noto Sans KR', sans-serif; }
    h1 { color: #ffffff !important; font-weight: 700 !important; }
    h2, h3 { color: #f97316 !important; }

    /* 대화형 레이아웃 */
    .coach-container { display: flex; align-items: flex-start; gap: 25px; margin: 30px 0; padding: 25px; background: #181818; border-radius: 20px; border: 1px solid #333; }
    .coach-img { width: 180px; height: 180px; border-radius: 50%; border: 4px solid #f97316; object-fit: cover; box-shadow: 0 0 20px rgba(249, 115, 22, 0.4); }
    .speech-bubble { position: relative; background: #282828; padding: 25px; border-radius: 20px; color: #e0e0e0; font-size: 1.1rem; line-height: 1.8; flex-grow: 1; border: 1px solid #444; }
    .highlight-text { color: #f97316; font-weight: 700; }

    /* 지표 카드 */
    .metric-card { background: #282828; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #333; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f97316; margin: 10px 0; }
    .metric-label { font-size: 0.9rem; color: #b3b3b3; text-transform: uppercase; }
</style>
"""

st.set_page_config(page_title="VAM AI Vocal Coach", layout="wide")
st.markdown(ST_STYLE, unsafe_allow_html=True)

st.title("🎤 AI 보컬 코칭 분석 대시보드")

with st.sidebar:
    st.header("🎵 가창 분석 시작")
    uploaded_file = st.file_uploader("WAV 파일을 선택하세요", type=["wav"])
    user_id = st.text_input("사용자 ID", value="test_user_001")
    analyze_btn = st.button("분석 시작", type="primary", disabled=not uploaded_file)
    st.divider()
    st.caption("v1.6.0 - Pro Studio Edition")

if analyze_btn and uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        with st.status("AI 트레이너가 분석 중입니다...", expanded=True) as status:
            extractor = VocalFeatureExtractor(tmp_path)
            features = extractor.process_all()
            
            compressor = VocalSemanticCompressor()
            semantic_text = compressor.compress_to_semantic_text(features)
            
            prompt_path = os.path.join(BASE_DIR, "Feedback_Generator", "analysis_feedback_prompt.txt")
            coach = VocalCoachLLM(api_key=api_key, prompt_file_path=prompt_path)
            feedback = coach.generate_feedback(semantic_text)
            
            db_manager = DBManager()
            session_id = db_manager.save_vocal_data(features, user_id)
            db_manager.update_feedback(session_id, feedback)
            status.update(label="분석 완료!", state="complete", expanded=False)

        # 1. 상단: 보컬 코치 피드백 (대화형)
        st.markdown("### 🤖 VAM AI 트레이너의 코칭")
        col_img, col_bubble = st.columns([1, 4])
        with col_img:
            if img_base64:
                st.markdown(f'<img src="data:image/png;base64,{img_base64}" class="coach-img">', unsafe_allow_html=True)
            else:
                st.warning("이미지 없음")
        with col_bubble:
            formatted_feedback = feedback.replace("🎤 전반적인 인상", "<span class='highlight-text'>🎤 전반적인 인상</span>") \
                                         .replace("⏱️ 구간별 디테일 분석", "<br><br><span class='highlight-text'>⏱️ 구간별 디테일 분석</span>") \
                                         .replace("💡 맞춤형 연습 처방", "<br><br><span class='highlight-text'>💡 맞춤형 연습 처방</span>")
            st.markdown(f'<div class="speech-bubble">{formatted_feedback}</div>', unsafe_allow_html=True)

        st.divider()

        # 2. 중단: 주요 지표 카드
        st.markdown("### 📊 주요 발성 지표")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card"><div class="metric-label">평균 피치</div><div class="metric-value">{features.get("f0_mean", 0):.1f} Hz</div></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card"><div class="metric-label">성대 접촉(HNR)</div><div class="metric-value">{features.get("hnr_db", 0):.1f} dB</div></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card"><div class="metric-label">바이브레이션</div><div class="metric-value">{features.get("vibrato_rate_hz", 0):.1f} Hz</div></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card"><div class="metric-label">긴장도(Jitter)</div><div class="metric-value">{features.get("jitter_percent", 0):.2f} %</div></div>', unsafe_allow_html=True)

        # 3. 하단: 상세 시각화 그래프 (Pitch & Energy 복구)
        st.markdown("### 📈 퍼포먼스 분석 그래프")
        
        # Pitch Contour (F0)
        f0_data = [x if x > 0 else None for x in features['f0_contour']]
        fig_f0 = go.Figure()
        fig_f0.add_trace(go.Scatter(y=f0_data, mode='lines', name='Pitch (Hz)', line=dict(color='#f97316', width=3)))
        fig_f0.update_layout(title="Pitch Contour (F0)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
        st.plotly_chart(fig_f0, width='stretch')

        # Energy Level (RMS) - 복구된 부분
        fig_rms = go.Figure()
        fig_rms.add_trace(go.Scatter(y=features['rms_contour'], mode='lines', name='Energy (RMS)', line=dict(color='#fbbf24', width=2)))
        fig_rms.update_layout(title="Volume Contour (RMS)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=250)
        st.plotly_chart(fig_rms, width='stretch')

        # 4. 최하단: 전문 분석 로그 익스팬더
        st.divider()
        with st.expander("📝 전문 분석 데이터 로그 보기 (Semantic Log)"):
            st.code(semantic_text, language="markdown")

    except Exception as e:
        st.error(f"분석 중 오류 발생: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
else:
    st.info("사이드바에서 WAV 파일을 업로드하고 분석을 시작하세요.")