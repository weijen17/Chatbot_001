

from langchain_community.tools.tavily_search import TavilySearchResults
import re
import numpy as np

def search_tool(input):
    try:
        search = TavilySearchResults(max_results=3)
        results = search.invoke(input)
        return results
    except Exception as e:
        return [{"error": str(e)}]



def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# def retrieval_tool(df_data,l_entities,embedder, faiss_index, query, TOP_K=TOP_K,TOP_K2=TOP_K2):
#     if l_entities:
#         '|'.join(l_entities)
#         df_retain = df_data[df_data['title_body'].str.contains('|'.join(l_entities), case=False)].reset_index()
#         allowed_ids = df_data['index'].values.tolist()
#         q_emb = embedder.encode([query], convert_to_numpy=True)
#         q_emb = l2_normalize(q_emb)
#
#         selector = faiss.IDSelectorBatch(
#             len(allowed_ids),
#             np.array(list(allowed_ids), dtype="int64")
#         )
#
#         params = faiss.SearchParametersIVF()
#         params.sel = selector
#
#         distances, indices = faiss_index.search(q_emb, TOP_K, params)
#         return [data[i] for i in indices[0] if i != -1]
#     else:
#         q_emb = embedder.encode([query], convert_to_numpy=True)
#         distances, indices = faiss_index.search(q_emb, TOP_K)
#         return [data[i] for i in indices[0]]



def retrieval_tool(data,l_entities,embedder, faiss_index, query, TOP_K,TOP_K2):
    q_emb = embedder.encode([query], prompt_name="query", convert_to_numpy=True)
    q_emb = l2_normalize(q_emb)
    distances, indices = faiss_index.search(q_emb, TOP_K)
    res = [data[i] for i in indices[0]]
    if l_entities and res:
        pattern = '|'.join(re.escape(element) for element in l_entities)
        res = [i for i in res if re.search(pattern, i[-1])]
    return res[:TOP_K2]

