# @File    : 02_文档拆分器_5_HTMLHeaderTextSplitter.py
# @Author  : Kenny So
# @Date    : 2025/11/11 1:09
# @Version : 1.0
from langchain_text_splitters import HTMLHeaderTextSplitter

# 1.定义HTML文件
html_string = """
    <!DOCTYPE html>
    <html>
    <body>
        <div>
            <h1>欢迎来到广州！</h1>
            <p>广州是千年古城</p>
            <div>
                <h2>广州简介</h2>
                <p>广州是我国的一线城市</p>
                <h3>广州越秀区</h3>
                <p>越秀区是广州的政治中心</p>
            </div>
        </div>
    </body>
    </html>
    """

# 2. 用于指定要根据哪些HTML标签来分割文本
headers_to_split_on = [
    ("h1", "标题1"),
    ("h2", "标题2"),
    ("h3", "标题3"),
]

# 3. 定义HTMLHeaderTextSplitter分割器
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# 4. 分割器分割
html_header_splits = html_splitter.split_text(html_string)

# 5. 打印
for block in html_header_splits:
    print(block.page_content)
