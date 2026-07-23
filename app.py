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

st.warning("""
**⚠️ 嚴格保密警告 (Confidentiality Notice)：**
根據《調解條例》（第620章）第8條、PD 31/31.1 及政府調解規則，調解通訊與會議內容具嚴格保密特權。
請**絕對不要**輸入任何真實案件的當事人姓名、公司名稱、財務條款或具體爭議細節。

*註：本系統僅作程序指引及學術參考，並不構成正式法律意見。*
""")

# --- 2. 獲取 Token ---
github_token = st.secrets.get("GITHUB_TOKEN") or st.secrets.get("OPENAI_API_KEY")
if not github_token:
    st.error("❌ 找不到 Token：請在 Streamlit App Settings -> Secrets 中設定 `GITHUB_TOKEN` 或 `OPENAI_API_KEY`。")
    st.stop()

# --- 3. 知識庫初始化 (Cap 620 + Cap 631 + 2025政府調解規則 + PD31/31.1 + 官方簡介 + 外部守則) ---
@st.cache_resource(show_spinner="正在加載《調解條例》、2025政府調解規則、實務指示及守則知識庫...")
def initialize_knowledge_base():
    documents = []
    
    # 策略 A: 載入本地法例、實務指示及政府 2025 最新規則
    local_files = [
        "data/Cap620.md", 
        "data/PD31.md", 
        "data/PD31_1.md",
        "data/Mediation_Intro.md",
        "data/Legal_Framework.md",
        "data/Gov_Mediation_Rules_2025.md"  # 新增：2025 政府調解規則
    ]
    for file_path in local_files:
        try:
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())
        except Exception as e:
            st.toast(f"⚠️ 無法載入本地檔案 {file_path}: {e}")

    # 策略 B: 動態抓取外部官方與機構網頁 (司法機構 FAQ, HKMAAL, HKMC, HKIAC)
    urls = [
        "https://mediation.judiciary.hk/tc/mediation_faq.html",                       # 司法機構 FAQ (中文)
        "https://mediation.judiciary.hk/en/mediation_faq.html",                       # 司法機構 FAQ (英文)
        "https://www.hkmaal.org.hk/tc/HongKongMediationCode.php",                      # HKMAAL
        "https://www.mediationcentre.org.hk/tc/services/MediationRules.php",          # HKMC
        "https://hkiac.org/zh-hant/other-services/mediation/rules/hkiac-mediation-rules/", # HKIAC 中文
        "https://hkiac.org/other-services/mediation/rules/hkiac-mediation-rules/"            # HKIAC 英文
    ]
    for url in urls:
        try:
            web_loader = WebBaseLoader(url)
            documents.extend(web_loader.load())
        except Exception as e:
            st.toast(f"⚠️ 無法即時讀取外部網頁 ({url})，系統將主要基於已載入資料回答。")

    if not documents:
        st.error("知識庫內容空白，請檢查 data/ 目錄下的 Markdown 檔案。")
        st.stop()

    # 文本切塊 (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # 建立向量庫
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = initialize_knowledge_base()

# --- 4. 串接 API 與 System Prompt ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=github_token,
    base_url="https://models.inference.ai.azure.com",
    temperature=0
)

system_prompt = (
    "你是一個專為香港新手調解員提供程序指引的 AI 助手。"
    "你必須遵守以下嚴格管治規則：\n"
    "1. 你的回答【必須且只能】基於提供的 Context（香港法例第620章《調解條例》、第631章《道歉條例》、2025年《香港特別行政區政府調解規則》、實務指示 PD 31 / PD 31.1、司法機構調解 FAQ、HKMAAL 守則、HKMC 規則或 HKIAC 調解規則）。\n"
    "2. 如果 Context 中沒有明確答案，你必須回答：『根據現有知識庫，無法提供確切答案，建議諮詢資深調解員或參考律師意見。』絕對不允許憑空捏造法律條文或法律意見。\n"
    "3. 在回答中，必須明確引用資料來源（例如：『根據《香港特別行政區政府調解規則（2025年版）》第 6 條』或『根據《調解條例》第 8 條』等）。\n"
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

# --- 5. 對話介面 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("請輸入關於香港調解程序、政府2025調解規則、道歉條例或保密條款的問題..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("檢索《調解條例》、2025政府調解規則及實務指示中..."):
            try:
                answer = rag_chain.invoke(user_query)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"系統生成回答時發生錯誤：{e}")
