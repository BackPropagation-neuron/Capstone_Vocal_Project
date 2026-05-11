import os
from google import genai
from dotenv import load_dotenv

def find_my_models():
    # .env 파일에서 API 키 불러오기
    load_dotenv()
    API_KEY = os.environ.get("GEMINI_API_KEY")
    
    if not API_KEY:
        print("API 키를 찾을 수 없습니다.")
        return

    # 클라이언트 연결
    client = genai.Client(api_key=API_KEY)
    
    print("🔍 내 API 키로 사용할 수 있는 모델 목록을 검색합니다...\n")
    
    # 사용 가능한 모델 목록 출력
    try:
        for model in client.models.list():
            # generateContent(텍스트 생성) 기능을 지원하는 모델만 필터링
            if 'generateContent' in model.supported_actions:
                print(f"사용 가능: {model.name}")
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")

if __name__ == "__main__":
    find_my_models()