import os

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

# 加载配置文件
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

####################################
# 1.获取对话模型
chat_model = ChatOpenAI(
    model = "gpt-4o-mini"
)

messages = [
    SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
    HumanMessage(content="人工智能英文怎么说？"),
    AIMessage(content="AI"),
    HumanMessage(content="你叫什么名字"),
]
messages1 = [
    SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
    HumanMessage(content="很高兴认识你"),
    AIMessage(content="我也很高兴认识你"),
    HumanMessage(content="你叫什么名字"),
]
messages2 = [
    SystemMessage(content="我是一个人工智能助手，我的名字叫小智"),
    HumanMessage(content="人工智能英文怎么说？"),
    AIMessage(content="AI"),
    HumanMessage(content="你叫什么名字"),
]

# 2. 调用大模型
response = chat_model.invoke(messages2)

print(response.content)
