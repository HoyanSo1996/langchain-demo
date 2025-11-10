# @File    : 01_文档加载器_DocumentLoaders.py
# @Author  : Kenny So
# @Date    : 2025/11/10 19:38
# @Version : 1.0
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader, UnstructuredHTMLLoader, \
    UnstructuredMarkdownLoader, DirectoryLoader, PythonLoader

'''
一、加载 txt 文件
'''
# 1. 定义加载器
text_loader = TextLoader("asset/01-langchain-utf-8.txt", encoding="utf-8")
# text_loader = TextLoader("asset/01-langchain-gbk.txt", encoding="gbk")

# 2. 加载, 返回List列表(Document对象)
text_docs = text_loader.load()

# 3. 打印
# print(text_docs)
# print(text_docs[0].page_content)


##########################################################################
'''
二、加载 pdf 文件 (也可以加载在线文档)
'''
# 1. 定义加载器
pdf_loader = PyPDFLoader(file_path="asset/02-阿里巴巴Java开发手册.pdf")
# pdf_loader = PyPDFLoader(file_path="https://arxiv.org/pdf/2302.03803")

# 2. 加载, 返回List列表(Document对象)
pdf_docs = pdf_loader.load()
# pdf_docs = pdf_loader.load_and_split()  # 另一种加载方式, 底层默认使用了递归字符文本切分器

# 3. 打印
# print(pdf_docs)
# for doc in pdf_docs:
#     print(doc.page_content)


##########################################################################
'''
三、加载 csv 文件
    加载不了, 待修复
'''
# 1. 初始化加载器
excel_loader = CSVLoader(file_path="asset/03-load.csv", encoding="gbk")

# 2. 加载, 返回List列表(Document对象)
excel_docs = excel_loader.load()

# 3. 打印
# print(excel_docs)
# for doc in excel_docs:
#     print(doc.page_content)


##########################################################################
'''
四、加载 json 文件
'''
# 1. 初始化加载器
# (1) 加载方式一
json_loader = JSONLoader(
    file_path="asset/04-load.json",
    jq_schema=".",  # 直接提取完整的JSON对象（包括所有字段）
    text_content=False  # 保持原始 JSON 结构，将提取的数据转换为 JSON 字符串存入 page_content 字段中
)
# (2) 加载方式二
json_loader2 = JSONLoader(
    file_path="asset/04-load.json",
    jq_schema=".messages[].content"   # 遍历 messages[] 属性中所有元素, 从每一个元素中提取 content 字段封装到 Document 对象中
)
# (3) 加载方式三. 通过 jq 语法精准定位目标数据, 对 JSON 中的嵌套字段、数组元素进行提取.
# 提取 04-response.json 文件中嵌套在 data.items[] 里的 title、content 和其文本
json_loader3 = JSONLoader(
    file_path="asset/04-response.json",
    jq_schema=".data.items[]",  # 先定位到数组条目
    content_key='.title + "\\t" + .content',  # 再从条目中提取 content 字段
    is_content_key_jq_parsable=True  # 用 jq 解析 content_key
)

# 2. 加载
json_docs = json_loader3.load()

# 3. 打印
# print(json_docs)
# 加载方式一
# print(json_docs[0].page_content)
# 加载方式二，三
# for doc in json_docs:
#     print(doc.page_content)


##########################################################################
'''
五、加载 html 文件
'''
# 1. 初始化加载器
html_loader = UnstructuredHTMLLoader(
    file_path="asset/05-load.html",
    mode="elements",
    strategy="fast"
)

# 2. 加载
html_docs = html_loader.load()

# 3. 打印
print(html_docs)
for doc in html_docs:
    print(doc.page_content)


##########################################################################
'''
六、加载 md 文件
'''
# 1. 初始化加载器
md_loader = UnstructuredMarkdownLoader(
    file_path="asset/06-load.md",
    strategy="fast",
    mode="elements",   # 可选项, 可以进行精细化切割
)

# 2. 加载
md_docs = md_loader.load()

# 3. 打印 for doc in docs:
# print(md_docs)
# for doc in md_docs:
#     print(doc.page_content)


##########################################################################
'''
七、加载 文件夹
'''
# 1. 初始化加载器
directory_loader = DirectoryLoader(
    path="asset",
    glob="*.py",
    use_multithreading=True,
    show_progress=True,
    loader_cls=PythonLoader
)

# 2. 加载
directory_docs = md_loader.load()

# 3. 打印
print(md_docs)
for doc in directory_docs:
    print(doc.page_content)
