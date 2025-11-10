# @File    : 02_文档拆分器_2_RecursiveCharacterTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/11 0:15
# @Version : 1.0
from langchain_text_splitters import RecursiveCharacterTextSplitter

'''
RecursiveCharacterTextSplitter:
    递归字符文本切分器, 遇到分割. 默认情况下，它尝试按顺序切割以下字符: ["\n\n", "\n", " ", ""] 。
'''
# 1. 自定义文本
text = "LangChain 框架特性\n\n多模型集成(GPT/Claude)\n记忆管理功能\n链式调用设计。文档分析场景示例：需要处理PDF/Word等格式。"

# 2. 初始化文档分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,
    add_start_index=True,
 )

# 3. 分割文本
paragraphs = text_splitter.split_text(text)

# 4. 打印
# for para in paragraphs:
#     print(para)
#     print("-" * 8)


#########################################################
# 1. 打开.txt文件
with open("asset/08-ai.txt", encoding="utf-8") as f:
    state_of_the_union = f.read()  # 返回的是字符串

# 2. 定义RecursiveCharacterTextSplitter（递归字符分割器）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    # chunk_overlap=0,
    length_function=len
)

# 3.分割文本
texts = text_splitter.create_documents([state_of_the_union])  # 使用 create_documents() 方法, 传入字符串列表, 返回 Document 对象列表

# 4.打印分割文本
for text in texts:
    print(f"🔥{text.page_content}")


#########################################################
'''
有些书写系统没有单词边界，例如中文、日文和泰文。使用默认分隔符列表["\n\n", "\n", "  ", ""]分割文
本可能导致单词错误的分割。为了保持单词在一起，你可以自定义分割字符，覆盖分隔符列表以包含额外的标点符号。

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,  # 增加重叠字符
    separators=["\n\n", "\n", "。", "！", "？", "……", "，", ""],  # 添加中文标点
    length_function=len,
    keep_separator=True # 保留句尾标点（如 ……），避免切割后丢失语气和逻辑
)
'''