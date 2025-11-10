# @File    : 02_文档拆分器_1_CharacterTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/10 23:06
# @Version : 1.0
from langchain_text_splitters import CharacterTextSplitter

'''
CharacterTextSplitter : Split by character
'''
# 1. 自定义文本
text = "LangChain 是一个用于开发由语言模型驱动的应用程序的框架的。它提供了一套工具和抽象，使开发者能够更容易地构建复杂的应用程序。"

# 2. 自定义字符分割器
splitter = CharacterTextSplitter(
    chunk_size=30,  # 每块大小
    chunk_overlap=5,  # 块与块之间的重复字符数
    length_function=len,
    # separator="",  # 设置为空字符串时，表示禁用分隔符来分割
    separator="。",  # 按指定分割分来分割  (此时优先在分隔符处分割, 然后再考虑 chunk_size, 即 chunk_size 可能会失效)
    keep_separator=True  # chunk 中是否保留切割符
)

# 3. 分割文本
texts = splitter.split_text(text)

# 4.打印结果
for i, chunk in enumerate(texts):
    print(f"块 {i + 1}:长度：{len(chunk)}")
    print(chunk)
    print("-" * 50)

