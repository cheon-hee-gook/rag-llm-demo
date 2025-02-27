from fastapi import FastAPI, Query
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

# FastAPI 앱 생성
app = FastAPI()

# ChromaDB에서 벡터 검색 기능 설정
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
retriever = vectorstore.as_retriever()

# LLM 모델 설정
qa_pipeline = pipeline("text-generation", model="facebook/opt-350m")


def generate_response(prompt):
    """LLM을 활용하여 답변 생성"""
    result = qa_pipeline(prompt, max_length=256, do_sample=True, truncation=True)

    return result[0]["generated_text"]


@app.get("/query")
async def query_rag(question: str = Query(..., description="질문을 입력하세요")):
    """RAG 기반 질문 응답 API"""
    retrieved_docs = retriever.invoke(question)

    # 가장 관련성 높은 문서 2개만 사용
    context = "\n".join([doc.page_content for doc in retrieved_docs[:2]])

    # LLM에게 답변을 요약하도록 요청
    response = generate_response(f"""
        You are an AI assistant specialized in answering questions based on retrieved documents.
        Use the following context to answer the question concisely in 3-4 sentences.

        Context: {context}

        Question: {question}
        Answer:
        """)

    return {"question": question, "answer": response}
