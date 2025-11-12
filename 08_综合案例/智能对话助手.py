import os

import dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.tools import Tool, create_retriever_tool
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

###########################################################
"""
目标：
    将构建一个可以与多种不同工具进行交互的Agent：一个是本地数据库，另一个是搜索引擎。你能
    够向该 Agent 提问，观察它调用工具，并与它进行对话。
    
涉及的功能：
    - 使用语言模型，特别是它们的工具调用能力
    - 创建检索器以向我们的 Agent 公开特定信息
    - 使用搜索工具在线查找信息
    - 提供聊天历史，允许聊天机器人 “记住” 不同id过去的交互，并在回答后续问题时考虑它们。
"""
# 1. 初始化搜索工具
search = TavilySearch(max_results=1)
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索互联网上的信息"
)

# 2. 初始化RAG工具
embedding_model = OpenAIEmbeddings()
# 加载文档
web_loader = WebBaseLoader("https://baike.baidu.com/item/%E7%8C%AB/22261")
docs = web_loader.load()
# 分割文档
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["。"])
documents = splitter.split_documents(docs)
# 初始化向量数据库
vector = FAISS.from_documents(documents, embedding_model)
# 获取检索器
retriever = vector.as_retriever()
# 封装工具
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="baidu_search",
    description="搜索百度百科"
)
# 测试查询结果
# print(retriever_tool.invoke("猫咪的生长特性"))

# 3. 准备工具
tools = [search_tool, retriever_tool]

# - 语言模型调用工具 (让大模型反思RAG中查出来的结果是否正确)
chat_model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = chat_model.bind_tools(tools)
# messages = [HumanMessage(content="今天上海天气怎么样")]
# response = model_with_tools.invoke(messages)
# print(response)

# 4. 创建 Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
# print(prompt.messages)
agent = create_tool_calling_agent(chat_model, tools, prompt)
# 创建 Agent 执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# 测试
# print(agent_executor.invoke({"input": "猫的特征"}))
# print(agent_executor.invoke({"input": "今天上海天气怎么样"}))


# 5. 添加记忆
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


agent_with_chat_history = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

response = agent_with_chat_history.invoke(
    input={"input": "Hi，我的名字是Cyber"},
    config={"configurable": {"session_id": "123"}}
)
print(f"回答一：{response}")

response = agent_with_chat_history.invoke(
    input={"input": "我叫什么名字?"},
    config={"configurable": {"session_id": "123"}}
)
print(f"回答二：{response}")

response = agent_with_chat_history.invoke(
    input={"input": "我叫什么名字?"},
    config={"configurable": {"session_id": "321"}}
)
print(f"回答三：{response}")

