# @File    : 01_Function_Call模式.py
# @Author  : Kenny So
# @Date    : 2025/11/10 17:21
# @Version : 1.0
import os

import dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###########################################################
# 1.初始化 搜索工具
search = TavilySearch(max_results=2)

# 2.初始化 LLM
llm = ChatOpenAI(
    # model="gpt-4",  # 太贵了
    model="gpt-4o-mini",
    temperature=0
)

# 3.定义提示词模版
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以回答问题并使用工具。"),
    ("placeholder", "{chat_history}"),  # 存储多轮对话的历史记录 如果你没有显式传入 chat_history，Agent 会默认将其视为空列表 []
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 4.定义记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 必须是此值，通过initialize_agent()的源码追踪得到
    return_messages=True
)

# 5.创建 Agent对象
agent = create_tool_calling_agent(llm, [search], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search], memory=memory, verbose=True)

# 6.测试对话
result1 = agent_executor.invoke({"input": "南京的天气是多少"})
print(result1)

print("==============================")
result2 = agent_executor.invoke({"input": "广州的呢"})
print(result2)

'''
如果删除 ChatPromptTemplate.from_messages() 中的 ("placeholder", "{chat_history}") 或者
是 AgentExecutor 构造方法中的 memory 参数, 会导致 Agent 失去上文的记忆，不知道第二问是什么意思。
继而回答: 你是想了解广州的什么信息呢？⽐如天⽓、旅游景点、⽂化活动、还是其他⽅⾯的内容？请具体说明⼀下，我会尽⼒帮你解答！
'''