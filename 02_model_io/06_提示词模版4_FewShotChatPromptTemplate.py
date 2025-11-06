# @File    : 06_提示词模版4_FewShotChatPromptTemplate.py
# @Author  : Kenny So
# @Date    : 2025/11/5 20:42
# @Version : 1.0
import os

import dotenv
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(
    model="gpt-4o-mini"
)

########################################
# 1.示例消息格式
examples = [
    {"input":"2🦜2","output":"4"},
    {"input":"2🦜3","output":"8"}
]

# 2.定义示例的消息格式提示词模版
example_prompt = ChatPromptTemplate.from_messages([
    ("human","{input}是多少?"),
    ("ai","{output}")
])

# 3.定义FewShotChatMessagePromptTemplate对象
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 4.输出完整提示词的消息模版
final_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '你是一个数学奇才'),
        few_shot_prompt,
        ('human', '{input}'),
    ]
)

print(chat_model.invoke(final_prompt.invoke("2🦜4")).content)  #  2🦜4 = 16