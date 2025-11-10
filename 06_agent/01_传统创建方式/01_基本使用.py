# @File    : 01_基本使用.py
# @Author  : Kenny So
# @Date    : 2025/11/9 23:41
# @Version : 1.0
import os

import dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

'''
# 传统方式(已过时):
# (1) 创建 Agent: 使用 AgentType 指定
# (2) 创建 AgentExecutor: initialize_agent()

缺点：不能使用提示词模版
'''

# 1.初始化搜索工具
search = TavilySearch(max_results=3)

# 2.封装Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索互联网上的信息"
)

# 3.初始化 llm
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 4.创建 AgentExecutor
agent_executor = initialize_agent(
    llm=llm,
    tools=[search_tool],
    # agent=AgentType.OPENAI_FUNCTIONS,  # （1）Function_call 模式, 结构化函数调用
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # （2）ReAct 模式, 自然语言推理
    verbose=True,
)

# 5.测试查询
result = agent_executor.invoke({"input": "昨天北京的天气怎么样？"})
print(f"查询结果: {result}")

