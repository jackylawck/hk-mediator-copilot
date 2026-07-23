import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. 管治與 UI 護欄 (Governance & UI Guardrails) ---
st.set_page_config(page_title="AI 調解員詢問站", page_icon="⚖️")
st.title("⚖️ 香港新手調解員 AI 詢問站 (PoC)")

# 強制顯示的免責聲明與保密警告 (AIGP Domain III/IV Guardrail)
st.warning("""
**⚠️ 嚴格保密警告 (Confidentiality Notice)：**
根據《調解條例》（第620章）第8條，調解通訊具嚴格保密特權。
請**絕對不要**輸入任何真實案件的當事人姓名、公司名稱、財務條款或具體爭議細節。

*註：本系統僅作程序指引及學術參考，並不構成正式法律意見。*
""")

# --- 2. 獲取 API Key ---
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ 找不到 API Key：請在 Streamlit Community Cloud 的 App Settings -> Secrets 中設定 `OPENAI_API_KEY`。")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

# --- 3. 知識庫初始化 (Knowledge Base Initialization) ---
@st.cache_resource(show_spinner="正在加載《調解條例》及外部守則知識庫...")
def initialize_knowledge_base():
    documents = []
    
    # 策略 A: 載入本地法例 (Cap 620)
    try:
        loader = TextLoader("data/Cap620.md", encoding="utf-8")
        documents.extend(loader.load())
    except Exception as e:
        st.error(f"無法載入本地 Cap620.md 檔案: {e}")

    # 策略 B: 動態抓取學會守則 (HKMAAL & HKMC)
    urls = [
        "https://www.hkmaal.org.hk/tc/HongKongMediationCode.php",
        "https://www.mediationcentre.org.hk/tc/services/MediationRules.php"
    ]
    for url in urls:
        try:
            web_loader = WebBaseLoader(url)
            documents.extend(web_loader.load())
        except Exception as e:
            # Graceful Degradation: 如果網絡抓取失敗，顯示提示但不中斷系統
            st.toast(f"⚠️ 無法即時讀取外部守則 ({url})，系統將主要基於《調解條例》回答。")

    if not documents:
        st.error("知識庫內容空白，請檢查 data/Cap620.md 檔案是否存在。")
        st.stop()

    # 文本切塊 (Chunking) 以提升 RAG 搜尋精確度
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 建立向量資料庫 (Vector Store)
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = initialize_knowledge_base()

# --- 4. System Prompt 設定 (防範幻覺與角色管治) ---
system_prompt = (
    "你是一個專為香港新手調解員提供程序指引的 AI 助手。"
    "你必須遵守以下嚴格管治規則：\n"
    "1. 你的回答【必須且只能】基於提供的 Context（香港法例第620章《調解條例》、HKMAAL守則或HKMC規則）。\n"
    "2. 如果 Context 中沒有明確答案，你必須回答：『根據現有知識庫，無法提供確切答案，建議諮詢資深調解員或參考律師意見。』絕對不允許憑空捏造法律條文或法律意見。\n"
    "3. 在回答中，必須明確引用資料來源（例如：『根據《調解條例》第8條(2)款...』）。\n"
    "4. 如果用戶在問題中提及看似真實的人名、公司名或機密數據，你必須拒絕回答，並提示其注意《調解條例》的保密條款。\n\n"
    "Context: {context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 建立 RAG 檢索與生成鏈
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Temperature=0 大幅降低幻覺風險
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- 5. 對話介面與用戶互動 (Chat Interface) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用戶輸入
if user_query := st.chat_input("請輸入關於香港調解程序或保密條例的問題..."):
    # 顯示用戶發問
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("檢索《調解條例》與守則中..."):
            try:
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"系統生成回答時發生錯誤：{e}")
