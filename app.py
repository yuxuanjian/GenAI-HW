import streamlit as st
from openai import OpenAI
import os
import io
import json
import fitz  
import docx
import uuid
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 頁面配置
st.set_page_config(page_title="My Chatbot", layout="wide")
st.title("My Chatbot")

# 注入 CSS 樣式 (保持緊湊排版)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    [data-testid="stSidebar"] button {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        min-height: 2rem !important;
        margin-top: -2px !important;
    }
    [data-testid="stSidebar"] .stWidgetLabel {
        margin-bottom: -10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 定義存檔路徑
SAVE_FILE = "chat_storage.json"

# 1. 存取與讀取邏輯
def save_data():
    data = {
        "folders": st.session_state.folders,
        "chats": st.session_state.chats,
        "current_chat_id": st.session_state.current_chat_id
    }
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.folders = data.get("folders", ["預設群組"])
                st.session_state.chats = data.get("chats", {})
                st.session_state.current_chat_id = data.get("current_chat_id", "")
        except Exception:
            pass

# 2. 初始化資料結構
if "folders" not in st.session_state:
    st.session_state.folders = ["預設群組"]
    st.session_state.chats = {}
    st.session_state.current_chat_id = ""
    load_data()
    
    if not st.session_state.chats:
        first_id = str(uuid.uuid4())
        st.session_state.chats = {
            first_id: {
                "title": "新對話", 
                "folder": "預設群組", 
                "messages": []
            }
        }
        st.session_state.current_chat_id = first_id

# 初始化全域變數
current_chunk = ""
chunk_idx_display = 0
uploaded_file_name = ""

# 3. 側邊欄：管理功能與設定
with st.sidebar:
    st.header("對話管理")
    
    with st.expander("管理功能", expanded=False):
        new_c_name = st.text_input("新對話名稱", value="新對話")
        if st.button("建立新對話", use_container_width=True):
            new_id = str(uuid.uuid4())
            st.session_state.chats[new_id] = {"title": new_c_name, "folder": "預設群組", "messages": []}
            st.session_state.current_chat_id = new_id
            save_data()
            st.rerun()
            
        st.markdown("---")
        new_f_name = st.text_input("建立新群組", placeholder="輸入後按回車")
        if new_f_name and new_f_name not in st.session_state.folders:
            st.session_state.folders.append(new_f_name)
            save_data()
            st.rerun()

    st.markdown("---")
    st.subheader("所有對話")
    for f in st.session_state.folders:
        with st.expander(f"群組: {f}", expanded=True):
            f_chats = {k: v for k, v in st.session_state.chats.items() if v["folder"] == f}
            if not f_chats:
                st.caption("尚無對話")
            
            for cid, cdata in f_chats.items():
                is_active = (cid == st.session_state.current_chat_id)
                btn_label = f"[目前] {cdata['title']}" if is_active else cdata['title']
                
                if st.button(btn_label, key=f"select_{cid}", use_container_width=True):
                    st.session_state.current_chat_id = cid
                    st.rerun()
                
                if st.button(f"刪除 {cdata['title']}", key=f"del_{cid}", use_container_width=True, type="secondary"):
                    del st.session_state.chats[cid]
                    if not st.session_state.chats:
                        new_id = str(uuid.uuid4())
                        st.session_state.chats[new_id] = {"title": "新對話", "folder": "預設群組", "messages": []}
                        st.session_state.current_chat_id = new_id
                    else:
                        if cid == st.session_state.current_chat_id:
                            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                    save_data()
                    st.rerun()

            # 刪除群組功能 (帶確認選項) - 強制狀態更新版
            if f != "預設群組":
                with st.popover(f"刪除 {f} 群組", use_container_width=True):
                    st.warning(f"要如何處理「{f}」內的對話？")
                    
                    # 選項 A：連同對話一起刪除
                    if st.button("全部刪除 (包含對話)", key=f"del_all_{f}", use_container_width=True, type="primary"):
                        # 1. 徹底刪除對話 (使用字典生成式，產生全新字典強迫 Streamlit 更新)
                        st.session_state.chats = {
                            cid: cdata for cid, cdata in st.session_state.chats.items() if cdata["folder"] != f
                        }
                        
                        # 2. 徹底移除群組名稱 (使用串列生成式，產生全新 List 強迫更新)
                        st.session_state.folders = [folder for folder in st.session_state.folders if folder != f]
                        
                        # 3. 安全檢查：若刪光了，補回預設對話
                        if not st.session_state.chats:
                            new_id = str(uuid.uuid4())
                            st.session_state.chats[new_id] = {"title": "新對話", "folder": "預設群組", "messages": []}
                            st.session_state.current_chat_id = new_id
                        elif st.session_state.current_chat_id not in st.session_state.chats:
                            st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
                        
                        save_data()
                        st.rerun()

                    # 選項 B：只刪群組，保留對話
                    if st.button("僅刪除群組 (對話移至預設)", key=f"keep_chats_{f}", use_container_width=True):
                        # 將對話移至預設群組
                        for cid in st.session_state.chats:
                            if st.session_state.chats[cid]["folder"] == f:
                                st.session_state.chats[cid]["folder"] = "預設群組"
                        
                        # 徹底移除群組名稱 (同樣使用串列生成式)
                        st.session_state.folders = [folder for folder in st.session_state.folders if folder != f]
                            
                        save_data()
                        st.rerun()

    st.markdown("---")
    st.header("設定面板")
    curr_id = st.session_state.current_chat_id
    if curr_id in st.session_state.chats:
        st.session_state.chats[curr_id]["title"] = st.text_input("重新命名目前對話", value=st.session_state.chats[curr_id]["title"], on_change=save_data)
        
        curr_f = st.session_state.chats[curr_id]["folder"]
        f_idx = st.session_state.folders.index(curr_f) if curr_f in st.session_state.folders else 0
        new_f_selection = st.selectbox("將目前對話移至群組", options=st.session_state.folders, index=f_idx)
        if new_f_selection != curr_f:
            st.session_state.chats[curr_id]["folder"] = new_f_selection
            save_data()
            st.rerun()

    st.markdown("---")
    selected_model = st.selectbox("選擇模型", ["qwen35-397b", "qwen35-4b"], index=0)
    system_prompt = st.text_area("System Prompt", value="你是一個專業助理。請用繁體中文回答")
    
    # 4. 檔案上傳與聰明切分邏輯 (帶重疊區間)
    uploaded_file = st.file_uploader("上傳參考檔案", type=['pdf', 'docx', 'txt', 'py', 'c', 'h', 'cpp', 'json', 'md'])
    
    if uploaded_file:
        uploaded_file_name = uploaded_file.name
        file_ext = uploaded_file_name.split('.')[-1].lower()
        try:
            raw_text = ""
            if file_ext == 'pdf':
                doc = fitz.open(stream=uploaded_file.getvalue(), filetype="pdf")
                raw_text = "".join([p.get_text() for p in doc])
            elif file_ext == 'docx':
                doc_file = docx.Document(io.BytesIO(uploaded_file.getvalue()))
                raw_text = "\n".join([p.text for p in doc_file.paragraphs])
            else:
                raw_text = uploaded_file.getvalue().decode("utf-8")

            # --- 差異化配額與重疊邏輯 ---
            if selected_model == "qwen35-4b":
                CHUNK_SIZE = 1500 # 4B 模型配額
                OVERLAP = 200      # 語意重疊
            else:
                CHUNK_SIZE = 6000 # 397B 模型配額
                OVERLAP = 500      
            
            chunks = []
            start = 0
            text_len = len(raw_text)
            
            while start < text_len:
                end = start + CHUNK_SIZE
                if end < text_len:
                    # 嘗試在換行處切割以保持整齊
                    last_newline = raw_text.rfind('\n', start, end)
                    if last_newline != -1 and last_newline > start:
                        end = last_newline + 1
                chunks.append(raw_text[start:end].strip())
                # 關鍵：下一段的起點是目前的終點減去重疊量
                start = end - OVERLAP
                if start >= end: start = end # 防止死迴圈
            
            total_chunks = len(chunks)
            if total_chunks > 1:
                st.info(f"檔案已針對 {selected_model} 切分為 {total_chunks} 個語意銜接片段。")
                chunk_idx = st.selectbox("選擇目前閱讀段落", range(total_chunks), format_func=lambda x: f"第 {x+1} 部分")
                current_chunk = chunks[chunk_idx]
                chunk_idx_display = chunk_idx + 1
            else:
                current_chunk = chunks[0] if chunks else ""
                chunk_idx_display = 1
            st.success(f"檔案 {uploaded_file_name} 讀取成功")
        except Exception as e:
            st.error(f"檔案解析失敗: {e}")

    temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    max_tokens = st.number_input("Max Tokens", 100, 100000, 25000)

# 初始化 API Client
client = OpenAI(api_key=os.getenv("CUSTOM_API_KEY"), base_url=os.getenv("CUSTOM_BASE_URL"))

# 5. 主畫面渲染與同步指令邏輯
curr_id = st.session_state.current_chat_id
if curr_id in st.session_state.chats:
    curr_chat = st.session_state.chats[curr_id]
    st.subheader(f"對話視窗: {curr_chat['title']}")

    for m in curr_chat["messages"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("請輸入訊息..."):
        curr_chat["messages"].append({"role": "user", "content": prompt})
        save_data()
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            full_sys = system_prompt + "\n\n【格式規範】數學公式請使用 LaTeX (行內 $，段落 $$)。"
            if current_chunk:
                # 在 Prompt 中同步物理序號，消除 4B 模型的理解障礙
                full_sys += (
                    f"\n\n【目前指定的資料片段內容】"
                    f"\n使用者目前選取並閱讀的是檔案《{uploaded_file_name}》的「第 {chunk_idx_display} 個語意切片」。"
                    f"\n請忽略內容中的章節編號與此序號的差異。當使用者說「這段」或「這部分」時，指的就是下方這段內容："
                    f"\n[片段開始]\n{current_chunk}\n[片段結束]"
                )

            # 依要求傳送「完整」歷史紀錄
            msgs = [{"role": "system", "content": full_sys}] + [{"role": m["role"], "content": m["content"]} for m in curr_chat["messages"]]

            try:
                response_placeholder = st.empty()
                full_response = ""
                stream = client.chat.completions.create(model=selected_model, messages=msgs, temperature=temp, max_tokens=max_tokens, stream=True)
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
                curr_chat["messages"].append({"role": "assistant", "content": full_response})
                save_data()
            except Exception as e:
                st.error(f"連線錯誤: {e}")