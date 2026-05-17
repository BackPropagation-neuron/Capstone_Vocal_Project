import os
import time
import numpy as np
from scipy.io import wavfile
from google import genai
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class VocalCoachChatSession:
    """
    하나의 인스턴스가 사용자의 '새로운 채팅방(New Chat)' 1개를 의미합니다.
    최초 분석 리포트 생성 후, 해당 맥락을 기억하며 대화를 이어나갈 수 있습니다.
    """
    def __init__(self, db_path="./vocal_rag_db"):
        load_dotenv()
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.model_id = 'gemini-2.5-flash'
        
        # 1. RAG 엔진 로딩 (채팅방이 열릴 때 한 번만 준비)
        print("로컬 RAG 지식 DB 로딩 중...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
            collection_name="vocal_textbooks"
        )
        
        # 2. 제미나이 대화(Chat) 세션 객체를 초기화 (상태 기억 공간)
        self.chat = None 

    def _retrieve_knowledge(self, query_text, k=3):
        docs = self.vectorstore.similarity_search(query_text, k=k)
        return "\n\n".join([f"[참고 문헌 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)])

    def start_initial_analysis(self, semantic_text, audio_file_path=None, user_request=None):
        """
        [첫 번째 턴] 최초 오디오를 분석하여 긴 메인 리포트를 뱉어냅니다.
        """
        # 새로운 채팅 세션 생성 (모델이 이제부터의 대화를 기억하기 시작함)
        self.chat = self.client.chats.create(model=self.model_id)
        
        expert_context = self._retrieve_knowledge(semantic_text)
        
        # V2 프롬프트 템플릿 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file_path = os.path.join(current_dir, "analysis_prompt_v2.txt")
        
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
            
        safe_user_request = user_request if user_request else "특별한 요청사항 없음. 전반적인 발성 상태와 시급한 문제점을 분석해 주세요."
        
        # 첫 번째 프롬프트에는 사용자의 모든 데이터(RAG, 수치 등)를 영혼까지 끌어모아 줍니다.
        initial_prompt = prompt_template.format(
            semantic_text=semantic_text,
            expert_context=expert_context,
            user_request=safe_user_request
        )
        
        contents_to_send = [initial_prompt]
        
        if audio_file_path and os.path.exists(audio_file_path):
            print(f"제미나이에게 오디오 파일({audio_file_path})을 전송하고 있습니다...")
            uploaded_audio = self.client.files.upload(file=audio_file_path)
            time.sleep(2)
            contents_to_send.append(uploaded_audio)

        print("[턴 1] 오디오와 수치를 분석하여 최초 리포트를 생성합니다...")
        
        # generate_content 대신 chat.send_message를 사용합니다.
        response = self.chat.send_message(contents_to_send)
        return response.text

    def send_followup_message(self, text_message):
        """
        [두 번째 턴 이후] 사용자의 추가 질문에 답변합니다.
        오디오나 RAG 데이터를 다시 보낼 필요 없이 텍스트만 보내면 됩니다!
        """
        if self.chat is None:
            return "오류: 최초 분석이 진행되지 않았습니다. 먼저 분석을 시작해 주세요."
            
        print(f"사용자 질문: {text_message}")
        print("[후속 턴] 이전 맥락을 바탕으로 코치가 답변을 고민 중입니다...")
        
        # 이전 대화와 오디오, 수치 분석 결과를 모델이 모두 기억하고 있습니다.
        response = self.chat.send_message(text_message)
        return response.text

# ==========================================
# 실행 및 검증 블록 (채팅 시나리오 시뮬레이션)
# ==========================================
if __name__ == "__main__":
    def create_dummy_audio_file(filename="test_vocal.wav"):
        sr = 22050
        t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        audio_data = 0.5 * np.sin(2 * np.pi * 440 * t + 2 * np.sin(2 * np.pi * 5 * t))
        wavfile.write(filename, sr, audio_data.astype(np.float32))
        return filename

    test_audio_path = create_dummy_audio_file()
    
    dummy_semantic_text = """
    ### 1. 전반적인 음향 통계
    - HNR: 8.5 dB (기준치 15dB 대비 낮음)
    - Jitter: 2.5 % (불안정)
    ### 2. 구간별 통계 분석
    [1.0초 ~ 2.0초]
      - F0: 평균 440.0Hz
      - Formants: F1 350Hz / F2 1500Hz / F3 2200Hz (후반부로 갈수록 수치 급감)
    """
    dummy_user_request = "끝음 처리가 불안한데, 어떻게 고칠 수 있을지 집중해서 봐주세요."
    
    try:
        # 1. 채팅방 개설 (새로운 세션)
        chat_session = VocalCoachChatSession()
        
        print("\n" + "="*50)
        print("[대화 시작] 최초 분석 리포트")
        print("="*50)
        
        # 2. 첫 번째 분석 요청
        initial_report = chat_session.start_initial_analysis(
            semantic_text=dummy_semantic_text,
            audio_file_path=test_audio_path,
            user_request=dummy_user_request
        )
        print(initial_report)
        
        print("\n" + "="*50)
        print("[후속 질문 1]")
        print("="*50)
        
        # 3. 추가 질문 (상태 유지 확인)
        # "아까 알려준 립트릴 연습"이라고만 말해도 모델이 맥락을 이해합니다.
        followup_reply = chat_session.send_followup_message("아까 추천해주신 연습 방법 중에서, 립트릴 연습은 하루에 몇 분 정도 하는 게 좋을까요? 제가 목이 잘 쉬는 편이라서요.")
        print(followup_reply)
        
    finally:
        if os.path.exists(test_audio_path):
            os.remove(test_audio_path)
            print(f"\n임시 테스트 파일 삭제 완료.")