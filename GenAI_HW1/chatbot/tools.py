from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSuggestion:
    name: str
    purpose: str
    why_simple: str
    example_query: str


SIMPLE_TOOL_SUGGESTIONS = [
    ToolSuggestion(
        name="web_search",
        purpose="補外部即時資訊，例如新聞、公司資訊、規格或公開網站內容。",
        why_simple="輸入與輸出都很直接，容易 demo，也很適合做自動 routing。",
        example_query="幫我查這家公司最新的產品公告，並整理成三點。",
    ),
    ToolSuggestion(
        name="calculator",
        purpose="處理四則運算、百分比、成本估算、單位轉換。",
        why_simple="實作成本最低，幾乎沒有 UI 負擔，能立刻提升正確率。",
        example_query="把 15% 折扣後再加 5% 稅的價格算給我。",
    ),
    ToolSuggestion(
        name="current_time",
        purpose="回答現在時間、時區換算、截止時間提醒。",
        why_simple="容易實作也容易展示，還能配合任務規劃類問答。",
        example_query="台北現在幾點？距離 4/29 23:59 還有多久？",
    ),
    ToolSuggestion(
        name="document_lookup",
        purpose="從你上傳的講義、作業說明、PDF 內找指定內容。",
        why_simple="和你現有的檔案上傳功能最接近，很適合升級成真正的 retrieval tool。",
        example_query="幫我找出 HW2 對 tool use 的要求原文。",
    ),
    ToolSuggestion(
        name="page_summarizer",
        purpose="讀取網址後做摘要、重點整理或比較。",
        why_simple="可以建立在 web search 之上，demo 效果明顯。",
        example_query="整理這篇文章的三個重點，並說明和我的專案有什麼關係。",
    ),
]


def list_tool_suggestions() -> list[ToolSuggestion]:
    return SIMPLE_TOOL_SUGGESTIONS
