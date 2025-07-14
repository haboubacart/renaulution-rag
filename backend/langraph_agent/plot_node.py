from backend.langraph_agent.utils import GraphState
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import json
import ast


def plot_node(state: GraphState) -> GraphState:
    stock_data = state.get("stock_data")
    rag_result = state.get("rag_result")
    df = None

    # Cas 1 : données boursières
    if stock_data:
        df = pd.DataFrame.from_dict(stock_data, orient="index")

    # Cas 2 : rag_result en JSON string
    elif rag_result:
        try:
            if isinstance(rag_result, str):
                try:
                    # Essayons d'abord de parser en JSON pur
                    parsed = json.loads(rag_result)
                except json.JSONDecodeError:
                    # Si échec, tentons une conversion via ast.literal_eval
                    parsed = ast.literal_eval(rag_result)
            else:
                parsed = rag_result  # déjà un dict

            if isinstance(parsed, dict):
                data = {}
                for year, value_str in parsed.items():
                    try:
                        value = float(value_str.replace(",", "").replace(" ", ""))
                        date = f"{year}-01-01"
                        data[date] = value
                    except Exception as e:
                        print(f"Erreur de parsing valeur '{value_str}' pour {year} : {e}")
                        continue
                if data:
                    df = pd.DataFrame.from_dict(data, orient="index", columns=["Valeur"])
        except Exception as e:
            print(f"Erreur de parsing JSON ou eval : {e}")
            return state

    if df is None or df.empty:
        return state

    # Mise en forme
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Tracé
    plt.figure(figsize=(6.5, 3.7))
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)

    plt.title("Évolution extraite")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend()
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    state["graph_base64"] = f"data:image/png;base64,{image_base64}"
    return state