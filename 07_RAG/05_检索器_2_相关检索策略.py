import os

import dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

###########################################################
# 1.定义文档
documents = [
    Document(page_content="经济复苏：美国经济正在从疫情中强劲复苏，失业率降至历史低点。"),
    Document(page_content="基础设施：政府将投资1万亿美元用于修复道路、桥梁和宽带网络。"),
    Document(page_content="气候变化：承诺到2030年将温室气体排放量减少50%。"),
    Document(page_content=" 医疗保健：降低处方药价格，扩大医疗保险覆盖范围。"),
    Document(page_content="教育：提供免费的社区大学教育。"),
    Document(page_content="科技：增加对半导体产业的投资以减少对外国供应链的依赖。"),
    Document(page_content="外交政策：继续支持乌克兰对抗俄罗斯的侵略。"),
    Document(page_content="枪支管制：呼吁国会通过更严格的枪支管制法律。"),
    Document(page_content="移民改革：提出全面的移民改革方案。"),
    Document(page_content="社会正义：承诺解决系统性种族歧视问题。")
]

# 2.创建向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# 3.将文档向量化，添加到向量数据库索引中，得到向量数据库对象
db = FAISS.from_documents(documents, embeddings)

# 4.开始检索
'''
4.1 默认检索器使用相似性搜索
'''
# retriever = db.as_retriever(search_kwargs={"k": 3})  # 这里设置返回的文档数
# docs = retriever.invoke("经济政策")
# for i, doc in enumerate(docs):
#     print(f"结果 {i + 1}:\n{doc.page_content}\n")

'''
4.2 分数阈值查询
    注意: similarity_score_threshold 只会返回满足阈值分数的文档，不会获取文档的得分。如果想查询文档的得分
    是否满足阈值，可以使用向量数据库的 similarity_search_with_relevance_scores 查看。
'''
# retriever = db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={"score_threshold": 0.1}  # 只有相似度超过这个值才会召回
# )
# docs = retriever.invoke("经济政策")
# for doc in docs:
#     print(f"📌 内容: {doc.page_content}")

'''
4.3 MMR 搜索
'''
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"fetch_k":2}
)
docs = retriever.invoke("经济政策")
print(docs)
for doc in docs:
    print(f"📌 内容: {doc.page_content}")
