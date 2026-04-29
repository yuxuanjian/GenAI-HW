# HW2 My Very Powerful Chatbot

這是放在 `GenAI_HW2/hw2_chatbot_v2` 的 HW2 實作版本，保留 `documents` 資料夾不動。

## 已完成的能力

1. 長期記憶：使用 `pinned memory + 記住/忘記指令 + memory summary + recent history + structured memory JSON store`。Structured memory 會分成 preferences / projects / facts / episodes / procedures，回答前檢索注入，回答後保守更新。
2. 多模態：可上傳圖片並以 vision model 理解圖片。
3. 自動路由：依輸入內容在小模型 / 大模型 / vision model 間切換。
4. Tool use：
   - `web_search`
   - `calculator`
   - `document_retrieval`
   - 以上工具預設優先透過 `stdio MCP` server/client 呼叫，若本機尚未安裝 `mcp` 套件則退回本地函式
   - 若同時使用圖片或文件與 `web_search`，會先做 query planning：從圖片 OCR/外觀線索或文件片段產生更適合搜尋的查詢，再送進搜尋工具
5. 文件知識庫：可上傳 PDF / DOCX / TXT / MD / 程式碼文字檔，建立本地 chunk library；檢索採 `BM25`，若設定 `EMBEDDING_MODEL` 則會對候選片段做 embedding rerank。
6. 資產隔離：文件庫與圖片上下文都綁定在單一對話，不會跨 chat 共用；但同一個 chat 內可重複引用先前上傳的圖片。

## 專案結構

```text
hw2_chatbot_v2/
├── app.py
├── chatbot/
│   ├── chat_view.py
│   ├── config.py
│   ├── documents.py
│   ├── llm.py
│   ├── memory.py
│   ├── memory_store.py
│   ├── models.py
│   ├── prompts.py
│   ├── routing.py
│   ├── sidebar.py
│   ├── state.py
│   ├── storage.py
│   └── tools/
│       ├── calculator.py
│       ├── retrieval.py
│       └── web_search.py
└── data/
```

## 如何啟動

1. 安裝套件

```bash
pip install -r requirements.txt
```

2. 建立 `.env`

```bash
cp .env_example .env
```

3. 填入 `CUSTOM_API_KEY` 與 `CUSTOM_BASE_URL`

若你要讓 `web_search` 優先用 Google Serper，可額外填 `SERPER_API_KEY`。
若沒填，系統會退回 DuckDuckGo HTML 搜尋。
若你要測圖片理解，`Vision Model` 必須填成 backend 真正支援多模態輸入的模型；若填的是一般文字模型，圖片能力會很差或近乎無效。

4. 啟動

```bash
streamlit run app.py
```
