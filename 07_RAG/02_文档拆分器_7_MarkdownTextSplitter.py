# @File    : 02_文档拆分器_7_MarkdownTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/11 1:09
# @Version : 1.0
from langchain_text_splitters import MarkdownTextSplitter

# 1. 定义文本
# Tip: 使用大字符串时, 文本内容要顶住定格, 不能用 tab 将调格式, 否则分割出来的效果有问题
markdown_text = """
# 一级标题\n
这是一级标题下的内容\n\n
## 二级标题\n
- 二级下列表项1\n
- 二级下列表项2\n
"""

# 2. 关键步骤：直接修改实例属性
splitter = MarkdownTextSplitter(chunk_size=30, chunk_overlap=0)
splitter._is_separator_regex = True   # 强制将分隔符视为正则表达式

# 3. 分割
docs = splitter.create_documents(texts=[markdown_text])

# 4. 打印
for i, doc in enumerate(docs):
    print(f"\n🔍 分块 {i + 1}:")
    print(doc.page_content)
