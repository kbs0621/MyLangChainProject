import os
from dotenv import load_dotenv
from langchain_community.utilities.tavily_search import TAVILY_API_URL
import re
from langchain_community.document_loaders import TextLoader, WikipediaLoader
from langchain_core.documents import Document
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_core.runnables import RunnableLambda, RunnableConfig, chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
from textwrap import dedent
from typing import List

# ======================================
# 환경 변수 로드
# ======================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

print(OPENAI_API_KEY[:2], UPSTAGE_API_KEY[:2], TAVILY_API_KEY[:2])

# ======================================
# 로컬 카페 메뉴 데이터 로드
# ======================================
loader = TextLoader("../data/cafe_menu_data.txt", encoding="utf-8")
documents = loader.load()
print("==> 1. cafe_menu.txt 로드 완료")

# --- 메뉴 항목별로 Document로 분할 ---
def split_menu_items(document):
    pattern = r'(\d+\.\s.*?)(?=\n\n\d+\.|$)'
    menu_items = re.findall(pattern, document.page_content, re.DOTALL)
    menu_documents = []
    for i, item in enumerate(menu_items, 1):
        menu_name = item.split('\n')[0].split('.', 1)[1].strip()
        menu_doc = Document(
            page_content=item.strip(),
            metadata={
                "source": document.metadata['source'],
                "menu_number": i,
                "menu_name": menu_name
            }
        )
        menu_documents.append(menu_doc)
    return menu_documents

menu_docs = split_menu_items(documents[0])
print(f"==> 2. 총 {len(menu_docs)}개의 메뉴 항목으로 분할 완료")

# ======================================
# 임베딩 모델 설정
# ======================================
embeddings_model = UpstageEmbeddings(model="solar-embedding-1-large")
print("==> 3. Upstage 임베딩 모델 준비 완료")

# ======================================
# Chroma 벡터 DB 생성 및 저장
# ======================================
db_dir = "./db/cafe_db"
cafe_db = Chroma.from_documents(
    documents=menu_docs,
    embedding=embeddings_model,
    persist_directory=db_dir
)
print("==> 4. Chroma 벡터 DB 생성 및 저장 완료")

# ======================================
# 검색기(Retriever) 테스트
# ======================================
retriever = cafe_db.as_retriever(search_kwargs={"k": 2})
retrieved_docs = retriever.invoke("라떼에 대해 알려줘")
print("\n--- 검색기 테스트 결과 ---")
for doc in retrieved_docs:
    print(doc.metadata)
print("==> 5. 검색기 테스트 완료")

# ======================================
# 도구 정의
# ======================================


# --- (1) Tavily 검색 도구 ---
@tool
def tavily_search_func(query: str) -> str:
    """
    최신 정보나 DB에 없는 정보를 인터넷에서 검색.
    예: 최신 커피 트렌드, 특정 카페 위치 정보 등.
    """
    tavily_search = TavilySearchResults(max_results=3)
    docs = tavily_search.invoke(query)

    formatted_docs = "\n---\n".join([
        f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
        for doc in docs
    ])

    if formatted_docs:
        return formatted_docs
    return "관련 정보를 찾을 수 없습니다."

print("==> 6-1. tavily_search_func 정의 완료")

# --- (2) Wikipedia 요약 도구 ---
def search_wiki_func(input_data: dict) -> List[Document]:
    wiki_loader = WikipediaLoader(query=input_data["query"], load_max_docs=2, lang="ko")
    return wiki_loader.load()

summary_prompt = ChatPromptTemplate.from_template(
    "다음 텍스트를 간결하게 요약해 주세요:\n\n{context}\n\n요약:"
)

llm = ChatUpstage(model="solar-pro", temperature=0.1)

summary_chain = (
    {"context": RunnableLambda(search_wiki_func)}
    | summary_prompt | llm | StrOutputParser()
)

class WikiSummarySchema(BaseModel):
    query: str = Field(..., description="위키피디아에서 검색할 주제")

wiki_summary = summary_chain.as_tool(
    name="wiki_summary",
    description=dedent("""
        일반 지식이나 배경 정보 필요 시 위키피디아에서 검색 후 요약.
        예: 커피의 역사, 음료 제조 방법 등.
    """),
    args_schema=WikiSummarySchema
)

print("==> 6-2. search_wiki_func 정의 완료")

# --- (3) 로컬 DB 검색 도구 ---
cafe_db = Chroma(
    embedding_function=embeddings_model,
    persist_directory=db_dir
)

@tool
def db_search_cafe_func(query: str) -> List[Document]:
    """
    로컬 카페 메뉴 데이터베이스에서 정보를 검색할 때 사용.
    메뉴의 가격, 재료, 설명 등에 대한 질문에 유용.
    """
    docs = cafe_db.similarity_search(query, k=4)
    if docs:
        return docs
    return [Document(page_content="관련 메뉴 정보를 찾을 수 없습니다.")]

print("==> 6-3. db_search_cafe_func 정의 완료")

# ======================================
# LLM에 도구 바인딩
# ======================================
tools = [db_search_cafe_func, tavily_search_func, wiki_summary]
llm_with_tools = llm.bind_tools(tools=tools)

print("==> 7. LLM 도구 바인딩 완료")
print("3개의 도구가 LLM에 성공적으로 바인딩되었습니다.")
print(f" - 바인딩된 도구: {[tool.name for tool in tools]}")


# ======================================
# 체인 구성
# ======================================
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 카페 메뉴와 음식에 대한 지식을 갖춘 AI 어시스턴트입니다. 사용자의 질문에 답하기 위해 주어진 도구를 적절히 활용하세요."),
    ("human", "{user_input}"),
    MessagesPlaceholder(variable_name="messages"),  # 도구 실행 결과 자리
])

llm_chain = prompt | llm_with_tools

@chain
def cafe_tool_chain(user_input: str, config: RunnableConfig):
    # 초기 입력
    input_ = {"user_input": user_input, "messages": []}

    # LLM 호출 → 도구 사용 결정
    ai_msg = llm_chain.invoke(input_, config=config)

    # 도구 실행
    tool_msgs = []
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"▶️ 도구 호출: {tool_call['name']}({tool_call['args']})")
            if tool_call["name"] == "db_search_cafe_func":
                tool_output = db_search_cafe_func.invoke(tool_call, config=config)
            elif tool_call["name"] == "tavily_search_func":
                tool_output = tavily_search_func.invoke(tool_call, config=config)
            elif tool_call["name"] == "wiki_summary":
                tool_output = wiki_summary.invoke(tool_call, config=config)

            tool_msgs.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call['id']))

    # 도구 결과 포함 → 최종 답변 생성
    input_["messages"].extend([ai_msg, *tool_msgs])
    return llm_chain.invoke(input_, config=config)

print("==> 8. @chain 데코레이터를 사용한 도구 호출 체인 구현 완료!")

# ======================================
# 테스트 질문 실행
# ======================================
query = "카페라테의 가격과 특징은 무엇인가요?"
response = cafe_tool_chain.invoke(query)

print("\n==> 9. 최종 답변 ")
print(response.content)