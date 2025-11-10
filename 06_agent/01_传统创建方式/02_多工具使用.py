# @File    : 02_多工具使用.py
# @Author  : Kenny So
# @Date    : 2025/11/10 1:16
# @Version : 1.0
import os

import dotenv
from langchain.agents import initialize_agent, AgentType
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###################################################################
# 1. 初始化搜索工具
search = TavilySearch(max_results=3)
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索互联网上的信息"
)

# 2. 初始化计算工具
python_repl = PythonREPL()
cal_tool = Tool(
    name="Calculator",
    func=python_repl.run,
    description="用于执行数学计算，例如计算百分比变化"
)

# 3. 初始化 LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 4. 创建 AgentExecutor 执行器对象
agent_executor = initialize_agent(
    tools=[search_tool, cal_tool],
    llm=llm,
    # agent=AgentType.OPENAI_FUNCTIONS,  # （1）Function_call 模式, 结构化函数调用
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # （2）ReAct 模式, 自然语言推理
    verbose=True
)

# 5. 测试股票价格查询
result = agent_executor.invoke({"input": "特斯拉当前股价是多少？比去年上涨了百分之几？"})
print(f"查询结果: {result}")
