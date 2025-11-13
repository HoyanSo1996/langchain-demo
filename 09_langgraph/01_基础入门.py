import datetime
import os

import dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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
agent = create_react_agent(
    model=chat_model,
    tools=[],
    prompt="You are a helpful assistant"
)

# 一次性输出
# res = agent.invoke({
#     "messages": [
#         {"role": "user", "content": "你是谁?"}
#     ]
# })
# print(res)

# 流式输出
res = agent.stream(
    {"messages" : [{"role": "user", "content": "你是谁?"}]},
    stream_mode="messages"
)
for chunk in res:
    print(chunk)
    print("\n")