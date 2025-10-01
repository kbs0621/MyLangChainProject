import os
from dotenv import load_dotenv
import re
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 환경 변수 로드
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(OPENAI_API_KEY[:2])


# --- 사전 준비: LLM 및 벡터 DB 로드 ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# 생성한 DB(cafe_db) 로드
try:
    cafe_db = FAISS.load_local("./db/cafe_db", embeddings, allow_dangerous_deserialization=True)
except RuntimeError as e:
    print("DB 로드 실패! 먼저 problem_4_1.ipynb를 실행하여 DB를 생성해주세요.")
    print(e)
    exit()

print("==> 0. 사전 준비 LLM 및 벡터 DB 로드 완료")
# --- LangGraph 구현 ---

# 1. 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]


print("==> 1. 상태 정의 완료")

# 2. 고급 정보 추출 함수
def extract_menu_info(doc: Document) -> dict:
    """Vector DB 문서에서 구조화된 메뉴 정보 추출 (정규표현식 수정)"""
    content = doc.page_content.strip()

    menu_match = re.search(r'^\d+\.\s*(.*)', content)
    price_match = re.search(r'•\s*가격:\s*(.*)', content)
    description_match = re.search(r'•\s*설명:\s*(.*)', content) # re.DOTALL 제거

    return {
        "name": menu_match.group(1).strip() if menu_match else "메뉴 정보 없음",
        "price": price_match.group(1).strip() if price_match else "가격 정보 없음",
        "description": description_match.group(1).strip() if description_match else "설명 없음"
    }

print("==> 2. 고급 정보 추출 함수 정의 완료")

# 3. 노드(Node) 함수 정의
def classify_query(state: State):
    user_message = state["messages"][-1].content
    
    if "가격" in user_message or "얼마" in user_message:
        return "price_query"
    elif "추천" in user_message or "어떤" in user_message or "뭐가" in user_message:
        return "recommendation_request"
    elif any(keyword in user_message for keyword in ["메뉴", "뭐 있어", "종류", "알려줘", "궁금해"]):
        menu_keywords = ["아메리카노", "라떼", "카푸치노", "프라푸치노", "마키아토", "콜드브루", "티라미수"]
        if any(menu in user_message for menu in menu_keywords):
            return "menu_query"
        return "menu_query"
    else:
        return "general_conversation"
    

print("==> 3-1. classify_query 함수 정의 완료")

def cafe_menu_query(state: State):
    user_message = state["messages"][-1].content
    docs = cafe_db.similarity_search(user_message, k=4)
    if not docs:
        response_content = "죄송합니다, 해당 메뉴에 대한 정보를 찾을 수 없습니다."
    else:
        info = extract_menu_info(docs[0])
        response_content = f"**{info['name']}**에 대한 정보입니다. \n- **설명**: {info['description']} \n- **가격**: {info['price']}"
    return {"messages": [AIMessage(content=response_content)]}


print("==> 3-2. cafe_menu_query 함수 정의 완료")

def cafe_price_query(state: State):
    user_message = state["messages"][-1].content
    docs = cafe_db.similarity_search(f"{user_message} 가격", k=5) 
    if not docs:
        response_content = "죄송합니다, 가격 정보를 찾을 수 없습니다."
    else:
        info = extract_menu_info(docs[0])
        response_content = f"문의하신 **{info['name']}**의 가격은 **{info['price']}** 입니다."
    return {"messages": [AIMessage(content=response_content)]}


print("==> 3-3. cafe_price_query 함수 정의 완료")

def cafe_recommendation_request(state: State):
    user_message = state["messages"][-1].content
    docs = cafe_db.similarity_search(user_message, k=3)
    if not docs:
        docs = cafe_db.similarity_search("인기 있고 맛있는 메뉴", k=3)
    
    info_list = [extract_menu_info(doc) for doc in docs]
    response_parts = [f"- **{info['name']}**: {info['description']} (가격: {info['price']})" for info in info_list]
    response_content = "이런 메뉴는 어떤가요? \n" + "\n".join(response_parts)
    return {"messages": [AIMessage(content=response_content)]}


print("==> 3-4. cafe_recommendation_request 함수 정의 완료")


def cafe_general_conversation(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}



print("==> 3-5. cafe_general_conversation 함수 정의 완료")

# 4. 그래프(Graph) 구성
graph_builder = StateGraph(State)

graph_builder.add_node("menu_query", cafe_menu_query)
graph_builder.add_node("price_query", cafe_price_query)
graph_builder.add_node("recommendation_request", cafe_recommendation_request)
graph_builder.add_node("general_conversation", cafe_general_conversation)

graph_builder.set_conditional_entry_point(
    classify_query,
    {
        "menu_query": "menu_query",
        "price_query": "price_query",
        "recommendation_request": "recommendation_request",
        "general_conversation": "general_conversation",
    }
)

graph_builder.add_edge("menu_query", END)
graph_builder.add_edge("price_query", END)
graph_builder.add_edge("recommendation_request", END)
graph_builder.add_edge("general_conversation", END)

graph = graph_builder.compile()

print("==> 4. 그래프 구성 완료")


# 5. 테스트 구성
def test_conversation(query: str):
    print(f"==> 사용자: {query}")
    events = graph.stream({"messages": [HumanMessage(content=query)]})
    for event in events:
        if "messages" in event.get(list(event.keys())[0], {}):
            last_message = event.get(list(event.keys())[0])["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"==> AI: {last_message.content}")
    print("-" * 30)

# 테스트 실행
print("==> 5. 테스트 실행")
test_conversation("안녕!")
test_conversation("커피 추천해줘")
test_conversation("프라푸치노 가격이 얼마인가요?")
test_conversation("티라미수 궁금해")