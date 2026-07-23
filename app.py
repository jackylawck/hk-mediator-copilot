import streamlit as st
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# --- 1. 管治與 UI 護欄 (Governance & UI Guardrails) ---
st.set_page_config(page_title="AI 調解員詢問站", page_icon="⚖️")
st.title("⚖️ 香港新手調解員 AI 詢問站 (PoC)")

# 強制顯示的免責聲明與保密警告
st.warning("""
**⚠️ 嚴重警告 (保密原則)：**
請勿輸入任何真實案件當事人姓名、公司名稱、財務數據或具體案件細節。根據《調解條例》（第620章）第8條，調解通訊具有嚴格的保密特權。
本系統僅供學術討論及程序指引，並非提供正式法律意見。
""")

# --- 2. 獲取 API Key ---
# 在 Streamlit Community Cloud 部署時，請在 Secrets 設定 OPENAI_API_KEY
api_key = st.secrets.get("OPENAI_API_KEY") 
if not api_key:
    st.error("請在 Streamlit Secrets 中設定 OPENAI_API_KEY。")
    st.stop()
os.environ["OPENAI_API_KEY"] = api_key

# --- 3. 知識庫初始化 (Knowledge Base Initialization) ---
@st.cache_resource(show_spinner="正在載入法例與守則知識庫...")
def initialize_knowledge_base():
    documents = []
    
    # 策略 A: 載入本地法例 (Cap 620)
    try:
        loader = TextLoader("data/Cap620.md", encoding="utf-8")
        documents.extend(loader.load())
        print("成功載入本地 Cap 620")
    except Exception as e:
        print(f"無法載入 Cap 620: {e}")

    # 策略 B: 動態抓取學會守則 (HKMAAL & HKMC)
    urls = [
        "https://www.hkmaal.org.hk/tc/HongKongMediationCode.php",
        "https://www.mediationcentre.org.hk/tc/services/MediationRules.php"
    ]
    for url in urls:
        try:
            web_loader = WebBaseLoader(url)
            documents.extend(web_loader.load())
            print(f"成功抓取: {url}")
        except Exception as e:
            print(f"無法抓取 {url}: {e}")
            st.toast(f"無法讀取外部守則：{url}，系統將僅依靠已載入的法例運作。")

    # 文本切塊 (Chunking) 以適應 LLM Context Window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 建立向量資料庫 (Vector Store)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore.as_retriever()

retriever = initialize_knowledge_base()

# --- 4. System Prompt 設定 (防範幻覺與角色設定) ---
system_prompt = (
    "你是一個專為香港新手調解員提供指引的 AI 助手。"
    "你必須遵守以下嚴格規則："
    "1. 你只能根據提供的 Context（香港法例第620章、HKMAAL守則、HKMC規則）回答問題。"
    "2. 如果 Context 中沒有相關資訊，你必須回答：『根據現有知識庫，無法提供確切答案，建議諮詢資深調解員。』，絕不允許憑空捏造法律意見。"
    "3. 在回答中，必須明確引用資料來源（例如：『根據《調解條例》第X條』）。"
    "4. 如果用戶輸入了看似真實的案件資料（如人名、公司名），必須立即拒絕回答，並提醒保密原則。"
    "\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 建立檢索與生成鏈
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Temperature 設為 0 以減少幻覺
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 5. 對話介面與用戶互動 (Chat Interface) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示過往對話紀錄
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用戶輸入
if user_query := st.chat_input("請輸入關於調解程序或守則的問題..."):
    # 顯示用戶輸入
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 呼叫 RAG 模型生成答案
    with st.chat_message("assistant"):
        with st.spinner("檢索法例與守則中..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"系統發生錯誤，請稍後再試。錯誤詳情：{e}"
                st.error(error_msg)
