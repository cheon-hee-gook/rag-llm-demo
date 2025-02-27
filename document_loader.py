import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ChromaDB 클라이언트 생성
chroma_client = chromadb.PersistentClient(path="./chroma_db")


# 다양한 기술 문서 추가
def load_documents_and_store():
    documents = [
        {"content": "Python은 동적 타입을 지원하는 고급 프로그래밍 언어입니다.", "id": "1"},
        {"content": "FastAPI는 비동기 웹 프레임워크로, 빠르고 효율적입니다.", "id": "2"},
        {"content": "ChromaDB는 벡터 기반 검색을 지원하는 데이터베이스입니다.", "id": "3"},
        {"content": "Django는 ORM을 지원하는 Python 기반 웹 프레임워크입니다.", "id": "4"},
        {"content": "Flask는 Python으로 작성된 마이크로 웹 프레임워크입니다.", "id": "5"},
        {"content": "Spring Boot는 Java 기반의 웹 프레임워크로, 자동 설정을 제공합니다.", "id": "6"},
        {"content": "Node.js는 비동기 이벤트 기반의 JavaScript 런타임입니다.", "id": "7"},
        {"content": "MySQL은 오픈소스 관계형 데이터베이스(RDBMS)입니다.", "id": "8"},
        {"content": "PostgreSQL은 ACID 트랜잭션을 지원하는 고급 데이터베이스입니다.", "id": "9"},
        {"content": "Redis는 인메모리 데이터베이스로, 빠른 키-값 저장 기능을 제공합니다.", "id": "10"},
        {"content": "MongoDB는 NoSQL 기반의 문서 지향 데이터베이스입니다.", "id": "11"},
        {"content": "Kafka는 대용량 데이터 스트리밍을 위한 메시지 브로커 시스템입니다.", "id": "12"},
        {"content": "Kubernetes는 컨테이너 오케스트레이션 도구입니다.", "id": "13"},
        {"content": "Docker는 컨테이너 기반 가상화 플랫폼입니다.", "id": "14"},
        {"content": "TensorFlow는 머신러닝과 딥러닝을 위한 오픈소스 라이브러리입니다.", "id": "15"}
    ]

    # 문서를 나누어 벡터화하기 위해 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = [text_splitter.split_text(doc["content"]) for doc in documents]
    texts = [item for sublist in texts for item in sublist]  # 리스트 평탄화

    # 벡터 데이터 저장
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    vectorstore.add_texts(texts)

    print("✅ 문서 데이터가 성공적으로 저장되었습니다. (총 {}개)".format(len(documents)))


if __name__ == "__main__":
    load_documents_and_store()
