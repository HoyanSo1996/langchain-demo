# @File    : 02_文档拆分器_3_TokenTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/11 0:15
# @Version : 1.0
import tiktoken
from langchain_text_splitters import TokenTextSplitter, CharacterTextSplitter

'''
TokenTextSplitter
    按Token的数量分割 （而非字符或单词数），将长文本切分成多个小块。
'''
# 1.定义文本
text = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。人工智能是指通过计算机程序模拟人类智能的一门科学。自20世纪50年代诞生以来，人工智能经历了多次起伏。"

# 2.初始化 TokenTextSplitter
text_splitter = TokenTextSplitter(
    chunk_size=33,  # 最大 token 数为 32
    chunk_overlap=0,  # 重叠 token 数为 0
    encoding_name="cl100k_base",  # 使用 OpenAI 的编码器,将文本转换为 token 序列
)

# 3.开始切割
texts = text_splitter.split_text(text)

# 3.打印分割结果
# print(f"原始文本被分割成了 {len(texts)} 个块:")
# for i, chunk in enumerate(texts):
#     print(f"块 {i + 1}: 长度：{len(chunk)} 内容：{chunk}")
#     print("-" * 50)


##############################################
# 使用 CharactorTextSplitter 来按 token 切割
# 1. 定义文本
text2 = "人工智能是一个强大的开发框架。它支持多种语言模型和工具链。今天天气很好，想出去踏青。但是又比较懒不想出去，怎么办."

# 2. 定义通过Token切割器
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # 使用 OpenAI 的编码器
    chunk_size=18,
    chunk_overlap=0,
    separator="。",  # 指定中文句号为分隔符
    keep_separator=False,  # chunk中是否保留分隔符
)

# 3. 开始切割
texts2 = text_splitter.split_text(text2)
print(f"分割后的块数: {len(texts2)}")

# 4.初始化tiktoken编码器（用于Token计数）
encoder = tiktoken.get_encoding("cl100k_base")  # 确保与CharacterTextSplitter的encoding_name一致

# 5.打印每个块的Token数和内容
for i, chunk in enumerate(texts2):
    tokens = encoder.encode(chunk)
    # 现在encoder已定义
    print(f"块 {i + 1}: {len(tokens)} Token\n内容: {chunk}\n")
