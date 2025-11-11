import os

import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

##############################################################
"""
使用RAG给LLM灌输上下文数据

"""
# 1.自定义模版
prompt = ChatPromptTemplate.from_template("""
    请使用以下提供的文本内容来回答问题。仅使用提供的文本信息，如果文本中
    没有相关信息，请回答"抱歉，提供的文本中没有这个信息"。
    文本内容：{context}
    问题：{question}
    回答：
    """)

# 2.初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# 3.加载文档
loader = TextLoader("asset/10-test_doc.txt", encoding='utf-8')
documents = loader.load()
# 分割文档
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
texts = text_splitter.split_documents(documents)

# 4.创建向量存储
vectorstore = FAISS.from_documents(
    documents=texts,
    embedding=embedding_model
)

# 5.获取检索器
retriever = vectorstore.as_retriever()
docs = retriever.invoke("北京有什么著名的建筑？")

# 6.创建 Runnable 链  / 等价于 prompt.invoke(), chat_model.invoke(prompt)
chain = prompt | chat_model

# 7. 提问
result = chain.invoke(input={"question": "北京有什么著名的建筑？", "context": docs})
print("\n回答:", result.content)