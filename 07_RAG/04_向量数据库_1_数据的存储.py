import os

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

#############################################################
# 1. 初始化模型
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# 2. 初始化文档加载器
text_loader = TextLoader("asset/09-ai1.txt", encoding='utf-8')
document = text_loader.load()

# 3. 初始化文本拆分器
# splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splitter = SemanticChunker(
    embeddings=embed_model,
    breakpoint_threshold_type="percentile",  # 断点阈值类型：字面值["百分位数", "标准差", "四分位距", "梯度"] 选其一
    breakpoint_threshold_amount=10.0  # 断点阈值数量 (极低阈值 → 高分割敏感度)
)
# 拆分文档
docs = splitter.create_documents([doc.page_content for doc in document])

# 4. 将文档和数据存储到向量数据库中
db_path = "./chroma_db"
db = Chroma.from_documents(docs, embed_model, persist_directory=db_path)

# 5. 查询
query = "人工智能的核心技术都有啥?"
answer = db.similarity_search(query)
print(f"answer = {answer[0].page_content}")
