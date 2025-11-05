import os

import dotenv
from langchain_openai import ChatOpenAI, OpenAI

# 1. 设置模型对象入参 (硬编码)
# 非对话模型
# llm = OpenAI(
#     # model = "gpt-4o-mini",  # 默认使用 gpt-3.5-turbo 模型
#     api_key = "sk-Xxxxxxxxxxxxxxxxxxx",
#     base_url = "https://api.openai-proxy.live/v1"
# )
# response = llm.invoke("什么是 langchain?")
# print(response)

# 对话模型
# llm = ChatOpenAI(
#     model = "gpt-4o-mini",   # 默认使用 gpt-3.5-turbo 模型
#     api_key = "sk-Xxxxxxxxxxxxxxxxxxx",
#     base_url = "https://api.openai-proxy.live/v1"
# )
# response = llm.invoke("解释神经网络原理")
# print(response.content)


# 2. 使用 环境变量 配置
# llm = ChatOpenAI(
#     # model = "",
#     # api_key = os.environ["OPENAI_API_KEY"],  # 可以省略, 因为 os.environ["OPENAI_API_KEY"] 是默认值
#     # base_url = os.environ["OPENAI_BASE_URL"], # 可以省略, 因为 os.environ["OPENAI_BASE_URL"] 是默认值
# )
# response = llm.invoke("解释神经网络原理")
# print(response.content)


# 3. 使用 .env 文件配置 (推荐)
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

llm = ChatOpenAI(
    base_url = os.environ["OPENAI_BASE_URL"],
    api_key = os.environ["OPENAI_API_KEY"],
)
response = llm.invoke("解释神经网络原理")
print(response.content)
