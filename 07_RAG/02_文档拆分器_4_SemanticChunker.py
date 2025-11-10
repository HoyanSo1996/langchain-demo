# @File    : 02_文档拆分器_4_SemanticChunker.py
# @Author  : Kenny So
# @Date    : 2025/11/11 1:09
# @Version : 1.0
import os

import dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# 初始化 LLM
embed_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

###########################################################
'''
SemanticChunker：语义分块
    是 LangChain 中一种更高级的文本分割方法, 它超越了传统的基于字符或固定大小的分块方式,
    而是根据文本的语义结构进行智能分块, 使每个分块保持语义完整性, 从而提高检索增强生成(RAG)等应用的效果。
'''
# 1. 加载文本
with open("asset/09-ai1.txt", encoding="utf-8") as f:
    state_of_the_union = f.read()  # 返回字符串

# 2. 获取切割器
text_splitter = SemanticChunker(
    embeddings=embed_model,
    breakpoint_threshold_type="percentile",  # 断点阈值类型：字面值["百分位数", "标准差", "四分位距", "梯度"] 选其一
    breakpoint_threshold_amount=50.0  # 断点阈值数量 (极低阈值 → 高分割敏感度)
)

# 3.切分文档
docs = text_splitter.create_documents(texts=[state_of_the_union])

# 4. 打印
print(len(docs))
for doc in docs:
    print(f"🔍 文档 {doc}:")
