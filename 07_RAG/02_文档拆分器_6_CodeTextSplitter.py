# @File    : 02_文档拆分器_6_CodeTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/11 1:09
# @Version : 1.0

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from pprint import pprint

# 1. 定义要分割的python代码片段
# Tip: 使用大字符串时, 文本内容要顶住定格, 不能用 tab 将调格式, 否则分割出来的效果有问题
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")
def hello_world1():
    print("Hello, World1!")
"""

# 2. 定义递归字符切分器
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0
)

# 3. 文档切分
python_docs = python_splitter.create_documents(texts=[PYTHON_CODE])

# 4. 打印
pprint(python_docs)
for code in python_docs:
    print(code.page_content)
