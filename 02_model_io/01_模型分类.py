import os

import dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")


#######  核心代码 #######
# 1.LLMs(非对话模型)
llm = OpenAI()
str = llm.invoke("写一首关于春天的诗")
print(str)


#################
# 2.Chat Models(对话模型)
chat_model = ChatOpenAI(model="gpt-4o-mini")
messages = [
    SystemMessage(content="我是人工智能助手，我叫小智"),
    HumanMessage(content="你好，我是小明，很高兴认识你")]
response = chat_model.invoke(messages)  # 输入消息列表
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)


#################
# 3.Embedding Model(嵌入模型)
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
res1 = embeddings_model.embed_query('我是文档中的数据')
print(res1) # 打印结果：[-0.004306625574827194, 0.003083756659179926, -0.013916781172156334, ....