from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_upstage import ChatUpstage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_upstage import UpstageEmbeddings

# .env 파일을 불러와서 환경 변수로 설정
load_dotenv()

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
print(UPSTAGE_API_KEY[30:])

print("==> 0. 문서 로딩 ")
file_path = "../data/콘텐츠분쟁해결_사례.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
print(f"  총 {len(documents)}페이지 로드 완료")


print("==> 1. 문서 분할 설정")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # 법률 사례는 1500자로 설정
    chunk_overlap=300,      # 사례 맥락 보존을 위해 300자로 설정
    separators=[
        "\n [사건개요]",  # 법률 문서 섹션 구분자
        "\n [쟁점사항]",  # 쟁점 부분 구분
        "\n [처리경위]",  # 처리 과정 구분
        "\n [처리결과]",  # 결과 부분 구분
        "\n\n", "\n", ".", " ",""
    ] 
)

chunks = text_splitter.split_documents(documents)
print(f"  {len(chunks)}개 청크 생성 완료")
print(f"  평균 청크 길이: {sum(len(chunk.page_content) for chunk in chunks) / len(chunks):.0f}자")
print(type(chunks[0]))

print("문서 분할 설정 완료")

print("==> 2.임베딩으로 변환")
pythonembeddings = UpstageEmbeddings(model="solar-embedding-1-large")

vectorstore = FAISS.from_documents(chunks, pythonembeddings)
print(f" 벡터스토어 생성 완료 ({len(chunks)}개 벡터)")
# 로컬 파일로 저장
vectorstore.save_local("python_db")

print("임베딩 변환 완료")

print(" ==> 3. 검색기 설정")
pythonretriever = vectorstore.as_retriever(
    search_type="similarity",        #  또는 "mmr" (다양성 고려시)
    search_kwargs={"k": 5}          #  상위 5개 관련 사례 검색
)

print("pythonretriever 설정 완료")

print(" ==> 4. LLM 설정")
pythonllm = ChatUpstage(
        model="solar-pro",
        base_url="https://api.upstage.ai/v1",
        temperature=0.2
    )

print("LLM 설정 완료")

print(" ==> 5. 법률 자문 프롬포트 작성")
pythonprompt_template = """
당신은 콘텐츠 분야 전문 법률 자문가입니다.
아래 분쟁 조정 사례들을 바탕으로 정확하고 전문적인 법률 조언을 제공해주세요.

관련 분쟁사례 : 
{context}

상담 내용 : {question}

답변 가이드라인 : 
1. 제시된 사례들을 근거로 답변하세요
2. 관련 법령이나 조항이 있다면 명시하세요
3. 비슷한 사례의 처리경위와 결과를 참고하여 설명하세요
4. 실무적 해결방안을 단계별로 제시하세요
5. 사례에 없는 내용은 "제시된 사례집에서는 확인할 수 없습니다."라고 명시하세요

전문 법률 조언: """

pythonprompt = PromptTemplate(
    template=pythonprompt_template,
    input_variables=["context", "question"]
)
print("법률 자문 프롬프트 설정 완료")


print(" ==> 6. QA 체인 생성")
pythonqa_chain = RetrievalQA.from_chain_type(
    llm=pythonllm,
    chain_type="stuff",
    retriever=pythonretriever,
    chain_type_kwargs={"prompt": pythonprompt},
    return_source_documents=True
)


print("QA 체인 생성 완료")

print(" ==> 7. 테스트 질문 작성")
pythontest_questions = [
    "온라인 게임에서 시스템 오류로 아이템이 사라졌는데, 게임회사가 복구를 거부하고 있습니다. 어떻게 해결할 수 있나요?",
    "인터넷 강의를 중도 해지하려고 하는데 과도한 위약금을 요구받고 있습니다. 정당한가요?",
    "무료체험 후 자동으로 유료전환되어 요금이 청구되었습니다. 환불 가능한가요?",
    "미성년자가 부모 동의 없이 게임 아이템을 구매했습니다. 환불받을 수 있는 방법이 있나요?",
    "온라인 교육 서비스가 광고와 다르게 제공되어 계약을 해지하고 싶습니다. 가능한가요?"
]

print(" ==> 8. 테스트 실행")
for i, question in enumerate(pythontest_questions, 1):
    print(f"\n【테스트 {i}/5】")
    print(f" 질문: {question}")
    print(" 답변 생성 중...")
    
    # RAG 실행
    result = pythonqa_chain.invoke({"query": pythontest_questions})
    answer = result["result"]
    source_docs = result["source_documents"]
    
    print(f"\n 답변:")
    print("-" * 50)
    print(answer)
    
    # 참조 문서 정보
    print(f"\n 참조 문서:")
    for j, doc in enumerate(source_docs[:3], 1):
        page = doc.metadata.get('page', 'N/A')
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"   {j}. 페이지 {page}: {preview}...")
    
    print("\n" + "-" * 40)