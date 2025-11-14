import datetime
import os

import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent, ToolNode

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

####################################################################################
#                                      分割                                         #
####################################################################################
# 初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")

# 初始化工具
@tool
def get_current_date():
    """获取今天日期"""
    return datetime.datetime.today().strftime("%Y-%m-%d")

agent = create_react_agent(
    model=chat_model,
    tools=[get_current_date],
    prompt="You are a helpful assistant"
)

# 一次性输出
# res = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "今天是几月几日?"}
#     ]
# })
"""
总共返回三条信息, 一条humanMessage, 两条AiMessage.
(1) 客户端请求LLM, 带上问题以及工具的描述信息
(2) LLM 综合判断问题, 并判断是否需要使用工具, 就会向客户端返回一个带有 tool_calls 工具调用信息的 AiMessage。
(3) 客户端根据工具调用信息，调用工具, 并将结果返回给大语言型, 大语言模型根据工具调用的结果, 生成最终的回答，
"""
# print(res)


###################################################
# 处理异常

@tool("divide_tool",return_direct=True)
def divide(a,b):
    """计算两个整数的除法
    Args:
        a:除数
        b:被除数
    """
    # 自定义除数
    if b == 1:
        raise ValueError("除数不能为1")
    return a/b

# 定义工具调用错误的处理函数
def handle_tool_error(error):
    """处理工具调用错误
    Args:
        error:工具调用错误
    """
    if isinstance(error,ValueError):
        return "除数1无意义，请重新输入一个被除数和除数"
    elif isinstance(error,ZeroDivisionError):
        return "除数0无意义，请重新输入一个被除数和除数"
    return f"工具调用错误：{error}"

# 定义工具集
tool_utils = ToolNode(
    tools=[divide],
    handle_tool_errors=handle_tool_error
)

# 初始化 agent
agent_with_error_handler = create_react_agent(
    model=chat_model,
    tools=tool_utils
)

res2 = agent_with_error_handler.invoke({
    # "messages" : [{"role":"user", "content": "10除以5等于多少？"}]
    "messages" : [{"role":"user", "content": "10除以5等于多少？"}]
})
print(res2)
