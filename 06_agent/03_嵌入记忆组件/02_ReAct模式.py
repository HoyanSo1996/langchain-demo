# @File    : 02_ReAct模式.py
# @Author  : Kenny So
# @Date    : 2025/11/10 17:21
# @Version : 1.0
import os

import dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
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
    model="gpt-4o-mini",
    temperature=0
)

# 3.定义提示词模版 (远程获取提示词模版)
# https://smith.langchain.com/hub/hwchase17/react-chat，这个模板是专为聊天场景设计的ReAct提示模板。
# 这个模板中已经有聊天对话键 chat_history 、agent_scratchpad
prompt = hub.pull("hwchase17/react-chat")
print(prompt)

# 4.定义记忆组件
memory = ConversationBufferMemory(
    memory_key="chat_history",  # 必须是此值, 可以与模版上对的上
    return_messages=True
)

# 5.创建 Agent对象
agent = create_react_agent(llm, [search], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[search], memory=memory, verbose=True)

# 6.测试对话
result1 = agent_executor.invoke({"input": "我的名字叫Bob"})
print(result1)

print("==============================")
result2 = agent_executor.invoke({"input": "请问我的名字叫什么？"})
print(result2)

