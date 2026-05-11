import os
from google import genai
from dotenv import load_dotenv

class VocalCoachLLM:
    def __init__(self, api_key=None, prompt_file_path="analysis_feedback_prompt.txt"):
        """
        새로운 google-genai 라이브러리를 사용하는 최신 클라이언트 초기화 방식입니다.
        api_key를 명시적으로 전달하거나, .env 파일에 GEMINI_API_KEY가 있으면 자동으로 인식합니다.
        """
        self.client = genai.Client(api_key=api_key)
        
        # 모델 이름 지정 (2.5 Flash 모델)
        self.model_id = 'gemini-2.5-flash' 
        self.prompt_file_path = prompt_file_path
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self):
        """텍스트 파일에서 프롬프트 템플릿을 읽어옵니다."""
        if not os.path.exists(self.prompt_file_path):
            raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {self.prompt_file_path}")
            
        with open(self.prompt_file_path, "r", encoding="utf-8") as file:
            return file.read()

    def generate_feedback(self, semantic_text):
        """
        Compressor에서 생성된 텍스트를 프롬프트에 주입하여 최종 피드백을 생성합니다.
        """
        # {diagnostic_text} 부분을 실제 분석 데이터로 치환합니다.
        final_prompt = self.prompt_template.format(diagnostic_text=semantic_text)
        
        try:
            # [수정됨] 새로운 라이브러리의 텍스트 생성 호출 방식
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=final_prompt
            )
            return response.text
        except Exception as e:
            return f"피드백 생성 중 오류가 발생했습니다: {e}"

# ===== 실행 예시 =====
if __name__ == "__main__":
    # 1. .env 파일에서 환경변수를 불러옵니다.
    load_dotenv()
    
    # 환경변수에서 키를 가져오거나, 못 찾으면 직접 입력할 수 있도록 합니다.
    API_KEY = os.environ.get("GEMINI_API_KEY") 
    
    if not API_KEY:
        print(".env 파일에 GEMINI_API_KEY가 설정되지 않았습니다.")
        exit()

    # 이전 단계(VocalSemanticCompressor)에서 생성된 결과물이라고 가정
    dummy_semantic_text = """
    ### 1. 전반적인 발성 상태 (Global Analysis)
    - [음색] 성대 접촉이 강해 밀도 있고 단단한 톤입니다.
    - [긴장도] 성대 진동(Jitter) 불규칙성이 감지됩니다. 발성 시 후두 긴장도가 높을 수 있습니다.
    
    ### 2. 시간대별 상세 진단 (Timeline Analysis)
    [00초~05초] 평균 220Hz 대역 | 음정 안정적, 다이내믹스 풍부함, 구강 공간 넓음
    [15초~20초] 평균 450Hz 대역 | 음정 흔들림, 구강 공간 좁음, 혀 위치 전진
    """
    
    try:
        # 클라이언트 객체 생성 및 실행
        coach = VocalCoachLLM(api_key=API_KEY)
        print("보컬 트레이너가 피드백을 작성 중입니다...\n")
        
        feedback = coach.generate_feedback(dummy_semantic_text)
        
        print("==================================================")
        print(feedback)
        print("==================================================")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")