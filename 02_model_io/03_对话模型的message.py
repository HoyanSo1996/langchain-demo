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

# 2.调用对话模型
system_message = SystemMessage(content = "你是一位英语教学方面的专家")
human_message = HumanMessage(content = "帮我指定一个英语六级学习的计划")
messages = [system_message, human_message]
response = chat_model.invoke(messages)

# 3.处理相应数据
print(response.content)
