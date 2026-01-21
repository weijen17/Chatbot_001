
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.assets.prompts import system_prompt__intention_recog, system_prompt__context_extraction,system_prompt__entity_extraction,system_prompt__name_search,system_prompt__search_content
from src.tools.tools_module import search_tool


def intent_recognition_agent(llm,input):
    response = llm.invoke([SystemMessage(content=system_prompt__intention_recog),
                           HumanMessage(content=input)])
    res = response.content.strip().lower()
    return res

def context_extraction_agent(llm,input):
    response = llm.invoke([SystemMessage(content=system_prompt__context_extraction),
                           HumanMessage(content=input)])
    res = response.content.strip().lower()
    return res

def entities_extraction_agent(llm,input):
    response = llm.invoke([SystemMessage(content=system_prompt__entity_extraction),
                           HumanMessage(content=input)])
    res = response.content.strip().lower()
    l_res = [i.strip() for i in res.split(',') if i not in ['NA','na']]

    return l_res

def similar_name_expansion_agent(llm,input):
    l_final_res = []
    for _ in input:
        results = search_tool(f'{_}的其他中文和英文昵称')
        response = llm.invoke([SystemMessage(content=system_prompt__name_search),
                               HumanMessage(content=f'Main name：{_}\n\n Search Result:\n{results}')])
        res = response.content.strip().lower()
        l_res = [_]+[i.strip() for i in res.split(',') if i not in ['NA','na']]
        l_final_res += l_res
    return l_final_res