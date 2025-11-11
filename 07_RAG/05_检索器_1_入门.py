import os

import dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###########################################################
# 1. 初始化文档加载器
text_loader = TextLoader(file_path='asset/09-ai1.txt', encoding="utf-8")

# 2. 加载文档
documents = text_loader.load()

# 3. 定义文本切割器
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# 4. 切割文档
docs = text_splitter.split_documents(documents)

# 5. 初始化嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 6. 将文档存储到向量数据库中
db = FAISS.from_documents(docs, embeddings)

# 7. 从向量数据库中获取检索器
retriever = db.as_retriever()

# 8. 使用检索器检索
docs = retriever.invoke("深度学习是什么？")  # 获取到的是整篇文档 (可能数据库中有很多片文档)

# 9. 获取结果
for doc in docs:
    print(f"⭐{doc}")