import os

import dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
'''
create_stuff_documents_chain 用于将多个文档内容合并成 LLM 处理（而不是分多次处理）
'''

# 定义提示词模板
prompt = PromptTemplate.from_template("如下文档{docs}中说，香蕉是什么颜色的？")

# 创建链
llm = ChatOpenAI(model="gpt-4o-mini")

chain = create_stuff_documents_chain(llm, prompt, document_variable_name="docs")

# 文档输入
docs = [
    Document(
        page_content="苹果，学名Malus pumila Mill.，别称西洋苹果、柰，属于蔷薇科苹果属的植" +
        "物。苹果是全球最广泛种植和销售的水果之一，具有悠久的栽培历史和广泛的分布范围。苹果的原始种" +
        "群主要起源于中亚的天山山脉附近，尤其是现代哈萨克斯坦的阿拉木图地区，提供了所有现代苹果品种" +
        "的基因库。苹果通过早期的贸易路线，如丝绸之路，从中亚向外扩散到全球各地。"
    ),
    Document(
        page_content="香蕉是白色的水果，主要产自热带地区。"
    ),
    Document(
        page_content="蓝莓是蓝色的浆果，含有抗氧化物质。"
    )
]

# 执行摘要
res = chain.invoke({"docs": docs})
print(res)