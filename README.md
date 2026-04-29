# GenAI Homework

This repository contains GenAI course homework implementations.

## Projects

- `GenAI_HW1/`: original Streamlit chatbot implementation.
- `GenAI_HW2/hw2_chatbot_v2/`: upgraded HW2 chatbot with MCP tools, routing, RAG/document retrieval, web search, image input, and long-term memory.

## Run HW2

```bash
cd GenAI_HW2/hw2_chatbot_v2
pip install -r requirements.txt
cp .env_example .env
streamlit run app.py
```

Fill `.env` with the required API keys and endpoint before running.
