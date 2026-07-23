import os
import streamlit as st
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. 管治與 UI 護欄 (Governance & UI Guardrails) ---
st.set_page_config(page_title="AI 調解員詢問站", page_icon="⚖️")
st.title("⚖️ 香港新手調解員 AI 詢問站 (PoC)")

# 強制顯示的免責聲明與保密警告
st.warning("""
**⚠️ 嚴格保密警告 (Confidentiality Notice)：**
根據《調解條例》（第620章）第8條，調解通訊具嚴格保密特權。
請**絕對不要**輸入任何真實案件的當事人姓名、公司名稱、財務條款或具體爭議細節。

*註：本系統僅作程序指引及學術參考，並不構成正式法律意見。*
""")

# --- 2. 獲取 GitHub Token (或 OpenAI API Key) ---
github_token = st.secrets.get("GITHUB_TOKEN") or st.secrets.get("OPENAI_API_KEY")
if not github_token:
    st.error("❌ 找不到 Token：請在 Streamlit App Settings -> Secrets 中設定 `GITHUB_TOKEN` 或 `OPENAI_API_KEY`。")
    st.stop()

# --- 3. 知識庫初始化 (加入 HKIAC 調解規則) ---
@st.cache_resource(show_spinner="正在加載《調解條例》、HKMAAL、HKMC 及 HKIAC 規則知識庫...")
def initialize_knowledge_base():
    documents = []
    
    # 策略 A: 載入本地法例 (Cap 620)
    try:
        loader = TextLoader("data/Cap620.md", encoding="utf-8")
        documents.extend(loader.load())
    except Exception as e:
        st.error(f"無法載入本地 Cap620.md 檔案: {e}")

    # 策略 B: 動態抓取外部學會與機構守則 (HKMAAL, HKMC, HKIAC 中英文版)
    urls = [
        "https://www.hkmaal.org.hk/tc/HongKongMediationCode.php",
        "https://www.mediationcentre.org.hk/tc/services/MediationRules.php",
        "https://hkiac.org/zh-hant/other-services/mediation/rules/hkiac-mediation-rules/",  # HKIAC 中文版
        "https://hkiac.org/other-services/mediation/rules/hkiac-mediation-rules/"            # HKIAC 英文版
    ]
    
    for url in urls:
        try:
            web_loader = WebBaseLoader(url)
            documents.extend(web_loader.load())
        except Exception as e:
            st.toast(f"⚠️ 無法即時讀取外部守則 ({url})，系統將主要基於已載入資料回答。")

    if not documents:
        st.error("知識庫內容空白，請檢查 data/Cap620.md 檔案是否存在。")
        st.stop()

    # 文本切塊 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 使用免費 HuggingFace Embeddings 模型
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = initialize_knowledge_base()

# --- 4. 串接 API (使用 GitHub Models 免費 API 或 OpenAI) ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=github_token,
    base_url="https://models.inference.ai.azure.com", # GitHub Models 端點
    temperature=0
)

# System Prompt 管治規範 (加入 HKIAC 引用要求)
system_prompt = (
    "你是一個專為香港新手調解員提供程序指引的 AI 助手。"
    "你必須遵守以下嚴格管治規則：\n"
    "1. 你的回答【必須且只能】基於提供的 Context（香港法例第620章《調解條例》、HKMAAL 守則、HKMC 規則或 HKIAC 調解規則）。\n"
    "2. 如果 Context 中沒有明確答案，你必須回答：『根據現有知識庫，無法提供確切答案，建議諮詢資深調解員或參考律師意見。』絕對不允許憑空捏造法律條文或法律意見。\n"
    "3. 在回答中，必須明確引用資料來源（例如：『根據《調解條例》第8條』或『根據 HKIAC 調解規則第X條』）。\n"
    "4. 如果用戶在問題中提及看似真實的人名、公司名或機密數據，你必須拒絕回答，並提示其注意《調解條例》的保密條款。\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. 對話介面 (Chat Interface) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("請輸入關於香港調解程序、HKIAC 規則或保密條例的問題..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("檢索《調解條例》、HKMAAL 及 HKIAC 規則中..."):
            try:
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"系統生成回答時發生錯誤：{e}")
