from  backend.langraph_agent.utils import GraphState
import yfinance as yf
import re

def finance_node(state: GraphState) -> GraphState:
    rag_context = state.get("rag_result", "")
    date_matches = re.findall(r"\d{4}-\d{2}-\d{2}", rag_context)
    dates = list(set(date_matches))
    if not dates:
        return state
    ticker_renault = yf.Ticker("RNO.PA").history(start="2020-01-01", end="2025-01-01")
    ticker_cac = yf.Ticker("^FCHI").history(start="2020-01-01", end="2025-01-01")

    stock_data = {}
    for date in dates:
        try:
            renault_price = ticker_renault.loc[date]["Close"]
            cac_price = ticker_cac.loc[date]["Close"]
            stock_data[date] = {
                "Renault": round(renault_price, 2),
                "CAC40": round(cac_price, 2)
            }
        except:
            continue
    return {"stock_data": stock_data}