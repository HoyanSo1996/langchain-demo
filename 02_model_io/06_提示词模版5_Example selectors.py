# @File    : 06_提示词模版5_SemanticSimilarityExampleSelector.py
# @Author  : Kenny So
# @Date    : 2025/11/5 20:42
# @Version : 1.0
import os

import dotenv
# from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
'''
前面FewShotPromptTemplate的特点是，无论输入什么问题，都会包含全部示例。在实际开发中，我
们可以根据当前输入，使用示例选择器，从大量候选示例中选取最相关的示例子集。

使用的好处：避免盲目传递所有示例，减少 token 消耗的同时，还可以提升输出效果。
'''

# 1. 定义示例组
examples = [
    {
        "question":"谁活得更久，穆罕默德·阿里还是艾伦·图灵?",
        "answer":"接下来还需要问什么问题吗？追问：穆罕默德·阿里去世时多大年纪？终极答案：穆罕默德·阿里去世时享年74岁。",
    },
    {
        "question":"craigslist的创始人是什么时候出生的？",
        "answer":"接下来还需要问什么问题吗？追问：谁是craigslist的创始人？终极答案：Craigslist是由克雷格·纽马克创立的。"
    },
    {
        "question":"谁是乔治·华盛顿的外祖父？",
        "answer":"接下来还需要问什么问题吗？追问：谁是乔治·华盛顿的母亲？终极答案：乔治·华盛顿的母亲是玛丽·鲍尔·华盛顿。"
    },
    {
        "question":"《大白鲨》和《皇家赌场》的导演都来自同一个国家吗？",
        "answer":"接下来还需要问什么问题吗？追问：《大白鲨》的导演是谁？终极答案：《大白鲨》的导演是史蒂文·斯皮尔伯格。"
    }
]

# 2. 定义嵌入模型
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

# 3. 定义示例选择器
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,  # 这是可供选择的示例列表
    embeddings_model,  # 这是用于生成嵌入的嵌入类，用于衡量语义相似性
    Chroma, # 这是用于存储嵌入并进行相似性搜索的 VectorStore 类， 也可以使用 FAISS,  pip install faiss-cpu
    k=1, # 这是要生成的示例数量
)

# 4.选择与输入最相似的示例
question = "玛丽·鲍尔·华盛顿的父亲是谁?"
selected_examples = example_selector.select_examples({"question": question})
print(f"与输入最相似的示例：{selected_examples}")