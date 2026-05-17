import os
import re
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# LangChain 구성 요소들
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def clean_pdf_text(text):
    """
    정규식(Regex)을 사용하여 PyPDF가 문서를 읽어올 때 발생하는 
    지저분한 줄바꿈과 공백 에러들을 깔끔하게 교정합니다.
    """
    if not text:
        return ""
    
    # 1. 단어 중간에 줄이 바뀐 경우 이어붙이기 (예: "vocaliza-\ntion" -> "vocalization")
    text = re.sub(r'-\s*\n\s*', '', text)
    
    # 2. 문장 한가운데 뜬금없이 들어간 줄바꿈을 공백으로 변경
    # (앞글자가 알파벳/쉼표이고, 뒷글자가 소문자 알파벳으로 시작하면 이어지는 문장으로 간주)
    text = re.sub(r'(?<=[a-zA-Z\,])\n(?=[a-z])', ' ', text)
    
    # 3. 3번 이상 연속된 줄바꿈은 2번(문단 분리)으로 압축
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. 불필요하게 긴 다중 공백을 단일 공백으로 압축
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def build_vocal_rag_db(data_folder="./Vocal_Docs", db_path="./vocal_rag_db"):
    # 1. 환경변수(API 키) 로드
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API 키를 찾을 수 없습니다. .env 파일을 확인하세요.")

    print(f"[{data_folder}] 폴더에서 문서를 읽어옵니다...")
    
    # 2. 문서 로드
    loader = PyPDFDirectoryLoader(data_folder)
    documents = loader.load()
    print(f"총 {len(documents)} 페이지 분량의 문서를 성공적으로 읽어왔습니다.")

    # ---------------------------------------------------------
    # Regex 텍스트 클리닝 단계
    # ---------------------------------------------------------
    print("정규식을 이용해 깨진 문장과 공백을 정제하고 있습니다...")
    for doc in documents:
        doc.page_content = clean_pdf_text(doc.page_content)
    # ---------------------------------------------------------

    # 3. 텍스트 나누기 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    print("문서를 AI가 소화하기 좋은 크기로 나누는 중입니다...")
    chunks = text_splitter.split_documents(documents)
    
    # ---------------------------------------------------------
    # 빈 데이터(Ghost Data) 완벽 필터링
    # ---------------------------------------------------------
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    print(f"전체 {len(chunks)} 조각 중, 유효한 텍스트가 있는 조각은 총 {len(valid_chunks)}개 입니다.")
    
    if len(valid_chunks) == 0:
        print("치명적 에러: 추출된 글자가 단 하나도 없습니다! PDF 상태를 확인하세요.")
        return
        
    print(f"[데이터 샘플 검증] 첫 번째 조각:\n{valid_chunks[0].page_content[:150]}...\n")
    # ---------------------------------------------------------

    # 4. 임베딩 및 ChromaDB 저장
    print("문서를 수학적 벡터로 변환하여 DB에 적재합니다. (약간의 시간이 소요될 수 있습니다...)")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}, # GPU가 세팅되어 있다면 'cuda'로 변경해도 됩니다.
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma.from_documents(
        documents=valid_chunks, # 필터링된 데이터 사용
        embedding=embeddings,
        persist_directory=db_path,
        collection_name="vocal_textbooks"
    )
    
    print(f"성공! 모든 지식이 [{db_path}] 폴더에 안전하고 깨끗하게 저장되었습니다.")

if __name__ == "__main__":
    build_vocal_rag_db()