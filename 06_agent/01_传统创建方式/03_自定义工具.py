# @File    : 03_自定义工具.py
# @Author  : Kenny So
# @Date    : 2025/11/10 1:48
# @Version : 1.0
import os

import dotenv
from langchain.agents import AgentType, initialize_agent
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###################################################################
# 1. 定义工具 - 计算器（要求字符串输入）
def simple_calculator(expression: str) -> str:
    """
    基础数学计算工具，支持加减乘除和幂运算
    参数:
    expression: 数学表达式字符串，如 "3+5" 或 "2**3"
    返回:
    计算结果字符串或错误信息
    """
    print(f"\n[工具调用] 计算表达式: {expression}")
    return str(eval(expression))

# 2. 创建工具对象
math_calculator_tool = Tool(
    name="Math_Calculator",  # 工具名称（Agent将根据名称选择工具）
    func=simple_calculator,  # 工具调用的函数
    description="用于数学计算，输入必须是纯数学表达式（如'3+5'或'3**2'表示平方）。不支持字母或特殊符号" # 关键：明确输入格式要求
)

# 3.初始化大模型
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

# 4. 初始化AgentExecutor（使用零样本React模式、增加超时设置）
agent_executor = initialize_agent(
    llm=llm,
    tools=[math_calculator_tool],  # 可用的工具列表
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # 简单指令模式
    # agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True   # 关键参数！在控制台显示详细的推理过程
)

# 5. 测试工具调用（添加异常捕获）
print("\n=== 测试：正常工具调用 ===")
response = agent_executor.invoke({"input": "计算3的平方"})  # 向Agent提问
print("最终答案:", response)
