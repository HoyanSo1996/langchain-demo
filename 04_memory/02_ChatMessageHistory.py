import os
import dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
'''
ChatMessageHistory是一个用于存储和管理对话消息的基础类，它直接操作消息对象（如 
HumanMessage, AIMessage 等），是其它记忆组件的底层存储工具。
'''
# 实例化
history = ChatMessageHistory()
# 添加 聊天记录
history.add_ai_message("我是一个无所不能的小智")
history.add_user_message("你好，我叫小明，请介绍一下你自己")
history.add_user_message("我是谁呢？")
# 返回存储的所有消息列表
print(history.messages)

# 对接 llm
llm = ChatOpenAI(model="gpt-4o-mini")
res = llm.invoke(history.messages)
print(res)