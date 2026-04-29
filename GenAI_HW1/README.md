這是一個基於Streamlit開發的客製化Chatbot網頁

## 核心功能
1. **挑選 LLM 模型**：支援在側邊欄自由切換(`qwen35-397b`) 與 (`qwen35-4b`)。
2. **自訂 System Prompt**：提供專屬文字區塊，可即時修改系統提示詞。
3. **自訂常用 API 參數**：可透過滑桿動態調整 `Temperature` 與 `Max Tokens`，即時影響模型回答的發散程度與長度。
4. **Streaming 串流輸出**。
5. **交談短期記憶**：保留對話歷史，讓模型能根據上下文進行追問與多輪對話。

## 其他功能
* **對話持久化與群組管理**：使用 `json` 進行本地存檔，重新整理網頁對話不遺失。支援新增/更名/刪除對話，以及將對話分類至自訂資料夾。
* **智慧檔案讀取與語意切分**：支援 `.pdf`, `.docx`, `.txt`, `.cpp`, `.py` 等多種格式。針對不同大小的模型（4B vs 397B）會自動進行不同長度的Overlap Chunking，避免 Context 溢出。

## 如何運行此專案

**1. 安裝相依套件**
```bash
pip install -r requirements.txt
```
或手動一個一個安裝requirements.txt列出的套件

**2. 將 .env_example 檔案重新命名為 .env，並填入您的 API KEY 與 Base URL**

**3. 啟動應用程式**
```
streamlit run app.py
```

註:如果助教需要我的url和api請通知我