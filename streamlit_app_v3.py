import os
import streamlit as st
import tempfile
import librosa
import soundfile as sf
import plotly.graph_objects as go
import base64
import numpy as np
from dotenv import load_dotenv

# 우리가 구축한 고도화된 V2 모듈 아키텍처 연동
from Vocal_Extractor.vocal_extractor import VocalFeatureExtractor
from Database_Manager.database_manager import DBManager
from Feedback_Generator_V2.semantic_compressor_v2 import VocalSemanticCompressorV2
from Feedback_Generator_V2.feedback_generator_v2 import VocalCoachChatSession


# ---------------------------------------------------------
# [V2 신규 유틸리티] 오디오 전처리 모듈
# ---------------------------------------------------------
def trim_silence_for_gemini(input_audio_path, output_audio_path="trimmed_temp.wav"):
    """
    오디오의 앞뒤 무음을 제거하여 Gemini 멀티모달 엔진이 가창 발성 구간에만 
    정밀하게 집중하여 음질을 파악할 수 있도록 도우미 파이프라인을 가동합니다.
    """
    y, sr = librosa.load(input_audio_path, sr=None)
    # top_db=30: 최대 음압 대비 -30dB 이하의 무음 구간을 계산하여 슬라이싱
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    sf.write(output_audio_path, y_trimmed, sr)
    return output_audio_path

# ---------------------------------------------------------
# 환경 변수 및 상대 경로 브랜딩 이미지 바인딩 (기존 로직 엄격 유지)
# ---------------------------------------------------------
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "vocal_coach.png")

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

img_base64 = get_base64_image(IMAGE_PATH)

# ---------------------------------------------------------
# 스포티파이 다크 디자인 테마 템플릿 정의 (CSS 컴포넌트 복원)
# ---------------------------------------------------------
ST_STYLE = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;500;700&display=swap');
    .stApp { background-color: #121212; color: #ffffff; font-family: 'Noto Sans KR', sans-serif; }
     Cinderella { display: none; }
    h1 { color: #ffffff !important; font-weight: 700 !important; }
    h2, h3 { color: #f97316 !important; }

    /* 대화형 전용 레이아웃 */
    .coach-container { display: flex; align-items: flex-start; gap: 25px; margin: 30px 0; padding: 25px; background: #181818; border-radius: 20px; border: 1px solid #333; }
    .coach-img { width: 180px; height: 180px; border-radius: 50%; border: 4px solid #f97316; object-fit: cover; box-shadow: 0 0 20px rgba(249, 115, 22, 0.4); }
    .speech-bubble { position: relative; background: #282828; padding: 25px; border-radius: 20px; color: #e0e0e0; font-size: 1.1rem; line-height: 1.8; flex-grow: 1; border: 1px solid #444; }
    .highlight-text { color: #f97316; font-weight: 700; }

    /* 통계 지표 대시보드 카드 */
    .metric-card { background: #282828; padding: 20px; border-radius: 12px; text-align: center; border: 1px solid #333; margin-bottom: 15px; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f97316; margin: 10px 0; }
    .metric-label { font-size: 0.9rem; color: #b3b3b3; text-transform: uppercase; }
</style>
"""

st.set_page_config(page_title="VAM AI Vocal Coach", layout="wide")
st.markdown(ST_STYLE, unsafe_allow_html=True)

st.title("🎤 AI 보컬 코칭 분석 대시보드")

# ---------------------------------------------------------
# Streamlit 세션 상태 스토리지 관리 (Rerun 대응 영속성 확보)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "coach_session" not in st.session_state:
    st.session_state.coach_session = None
if "db_session_id" not in st.session_state:
    st.session_state.db_session_id = None
if "vocal_features" not in st.session_state:
    st.session_state.vocal_features = None
if "vocal_semantic_text" not in st.session_state:
    st.session_state.vocal_semantic_text = None

# ---------------------------------------------------------
# 사이드바 컨트롤 컴포넌트 영역
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎵 가창 분석 시작")
    uploaded_file = st.file_uploader("WAV 파일을 선택하세요", type=["wav"])
    user_request = st.text_area("코치에게 특별히 피드백 받을 핵심 포인트 (선택)", placeholder="예: 후렴구 초고음 부분 파사지오 전환과 비브라토 깊이를 중점적으로 봐주세요.")
    analyze_button = st.button("분석 시작", type="primary", disabled=not uploaded_file)
    st.divider()
    st.caption("v2.0.0 - Pro Studio RAG Edition")

# ---------------------------------------------------------
# 오디오 업로드 및 핵심 연산 알고리즘 처리 (V2 Pipeline)
# ---------------------------------------------------------
if analyze_button and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_audio_path = tmp_file.name

    try:
        with st.status("AI 트레이너가 신호를 정렬하고 지식을 탐색 중입니다...", expanded=True) as status:
            # 1. 하이브리드 오디오 특징값 추출 (DSP Engine)
            extractor = VocalFeatureExtractor(tmp_audio_path)
            features = extractor.process_all()
            
            # [2-Step Gatekeeper: 1차 백엔드 최소 유효 보컬 신뢰도 검증]
            vocal_confidence = features.get('voiced_probs_mean', 0.0)
            if vocal_confidence < 0.20:
                st.error("🚨 가창 성대 진동 주파수가 감지되지 않았습니다. 악기 소리이거나 심한 노이즈 파일로 추정됩니다. 다시 가창해 주세요!")
                os.remove(tmp_audio_path)
                st.stop()
                
            # 2. 데이터 영구 저장 레이어 세션 발행
            db_manager = DBManager()
            st.session_state.db_session_id = db_manager.save_vocal_data(features, user_id="web_user_01")
            
            # 3. 데이터프레임 시계열 텍스트 압축화 기법 가동
            compressor = VocalSemanticCompressorV2()
            semantic_text = compressor.compress_to_semantic_text(features)
            
            # 4. 멀티모달 인퍼런스 대응 앞뒤 무음 음압 컷팅 적용
            trimmed_audio_path = trim_silence_for_gemini(tmp_audio_path)
            
            # 5. 독립 가창 발성학 RAG 검색 체인 탑재 및 제미나이 컨텍스트 초기화
            st.session_state.coach_session = VocalCoachChatSession(db_path="./vocal_rag_db")
            initial_feedback = st.session_state.coach_session.start_initial_analysis(
                semantic_text=semantic_text,
                audio_file_path=trimmed_audio_path,
                user_request=user_request
            )
            
            # 6. 생성된 최종 문헌 결합 리포트 데이터베이스 연동
            db_manager.update_feedback(st.session_state.db_session_id, initial_feedback)
            
            # 7. 휘발 방지를 위한 세션 상태 영구 마운트 수행
            st.session_state.vocal_features = features
            st.session_state.vocal_semantic_text = semantic_text
            st.session_state.messages = [{"role": "assistant", "content": initial_feedback}]
            
            # 서버 가상 자원 임시 청소
            os.remove(tmp_audio_path)
            if os.path.exists(trimmed_audio_path):
                os.remove(trimmed_audio_path)
                
            status.update(label="분석 파이프라인 연동 성공!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"백엔드 연산 처리 중 치명적 오류 발생: {e}")
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)

# ---------------------------------------------------------
# 상시 고정형 분석 렌더링 화면 대시보드 구축 (UI 복원부)
# ---------------------------------------------------------
if st.session_state.coach_session is not None and st.session_state.vocal_features is not None:
    features = st.session_state.vocal_features
    semantic_text = st.session_state.vocal_semantic_text
    initial_report = st.session_state.messages[0]["content"]

    # [파트 1] 상단: 커스텀 보컬 트레이너 메인 텍스트 리포트 출력
    st.markdown("### 🤖  AI 트레이너 맞춤 발성 교정 리포트")
    col_img, col_bubble = st.columns([1, 4])
    with col_img:
        if img_base64:
            st.markdown(f'<img src="data:image/png;base64,{img_base64}" class="coach-img">', unsafe_allow_html=True)
        else:
            st.warning("⚠️ 로컬 캐시 디렉터리 내 vocal_coach.png 리소스를 바인딩할 수 없습니다.")
    with col_bubble:
        # 가독성 하이라이팅 마크다운 가공 처리
        formatted_feedback = initial_report.replace("📌 [핵심 요약 리포트]", "<span class='highlight-text'>📌 [핵심 요약 리포트]</span>") \
                                           .replace("🔍 [상세 분석 리포트]", "<br><br><span class='highlight-text'>🔍 [상세 분석 리포트]</span>") \
                                           .replace("🎤 전반적인 인상", "<span class='highlight-text'>🎤 전반적인 인상</span>") \
                                           .replace("⏱️ 구간별 디테일 분석", "<br><br><span class='highlight-text'>⏱️ 구간별 디테일 분석</span>") \
                                           .replace("💡 맞춤형 연습 처방", "<br><br><span class='highlight-text'>💡 맞춤형 연습 처방</span>")
        st.markdown(f'<div class="speech-bubble">{formatted_feedback}</div>', unsafe_allow_html=True)

    st.divider()

    # [파트 2] 중단: 주요 4열 발성 지표 대시보드 카드 그리드 구현
    st.markdown("### 📊 주요 성구·음향 지표")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-card"><div class="metric-label">평균 주파수 (Pitch)</div><div class="metric-value">{features.get("f0_mean", 0):.1f} Hz</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-card"><div class="metric-label">성대 밀도 및 접촉 (HNR)</div><div class="metric-value">{features.get("hnr_db", 0):.1f} dB</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-card"><div class="metric-label">비브라토 변조율</div><div class="metric-value">{features.get("vibrato_rate_hz", 0):.1f} Hz</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card"><div class="metric-label">후두 긴장 주기성 (Jitter)</div><div class="metric-value">{features.get("jitter_percent", 0):.2f} %</div></div>', unsafe_allow_html=True)

    # [파트 3] 하단: Plotly 반응형 2단 시계열 곡선 렌더링
    st.markdown("### 📈 퍼포먼스 동적 트래킹 그래프")
    
    # Track 1: Pitch Contour (F0)
    f0_data = [x if x > 0 else None for x in features['f0_contour']]
    fig_f0 = go.Figure()
    fig_f0.add_trace(go.Scatter(y=f0_data, mode='lines', name='Pitch (Hz)', line=dict(color='#f97316', width=3)))
    fig_f0.update_layout(title="Pitch Flow (F0 Contour)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=320, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_f0, use_container_width=True)

    # Track 2: Volume Contour (RMS Energy)
    fig_rms = go.Figure()
    fig_rms.add_trace(go.Scatter(y=features['rms_contour'], mode='lines', name='Energy (RMS)', line=dict(color='#fbbf24', width=2)))
    fig_rms.update_layout(title="Volume Level (RMS Contour)", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white', height=240, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_rms, use_container_width=True)

    # [파트 4] 최하단: 공학 분석 원본 로그 데이터 축소 아코디언 창 복원
    st.divider()
    with st.expander("📝 시스템 분석 원문 자연어 로그 보기 (Statistical Semantic Log)"):
        st.code(semantic_text, language="markdown")

    # [파트 5] 지속형 가창 발성 대화식 챗 창 컴포넌트 분리 레이아웃 개설
    st.divider()
    st.markdown("### 💬 VAM 트레이너와 1:1 발성 학습 질의응답 (RAG)")
    
    # 메인 리포트(index 0)는 상단 HTML 말풍선 구역에 고정 노출되므로, 후속 꼬리 대화만 대화 로그에 뿌림
    for msg in st.session_state.messages[1:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("방금 분석된 가창 상태에 대해 궁금한 점을 코치에게 질문하세요!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("코치가 벡터 서적 자료를 참조하여 교정 훈련법을 도출 중입니다..."):
                response = st.session_state.coach_session.send_followup_message(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()
else:
    if not analyze_button:
        st.info("💡 가창 대시보드가 준비되었습니다. 왼쪽 사이드바에서 분석을 원하는 가창 녹음본(.wav)을 업로드하세요.")