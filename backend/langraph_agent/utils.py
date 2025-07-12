from typing import Optional, TypedDict

class GraphState(TypedDict):
    question: str
    route: Optional[str]
    rag_result: Optional[str]
    stock_data: Optional[dict]
    graph_base64: Optional[str]
    final_response: Optional[str]