# @File    : 03_文档嵌入模型.py
# @Author  : Kenny So
# @Date    : 2025/11/11 10:51
# @Version : 1.0
import os

import dotenv
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

#############################################################
# 1. 初始化模型
# embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 2. 句子向量化
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")

# 打印
# print(len(embedded_query))
# print(embedded_query)

# 3. 文档向量化
texts = [
    "Hi there!",
    "Oh, hello!",
    "What's your name?",
    "My friends call me World",
    "Hello World!"]
embeddings = embeddings_model.embed_documents(texts)

# 打印
for i in range(len(texts)):
    print(f"{texts[i]}:{embeddings[i][:3]}", end="\n")
