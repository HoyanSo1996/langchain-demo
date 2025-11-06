# @File    : 06_提示词模版2_ChatPromptTemplate.py
# @Author  : Kenny So
# @Date    : 2025/11/5 20:42
# @Version : 1.0
import os
import dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
# 1.实例化的方式 (两种方式：使用构造方法, from_message())
## 1.1 构造方式
chat_prompt_template = ChatPromptTemplate([
    ("system", "你是一个AI助手. 你的名字是 {name}."),
    ("human", "我的问题是{question}.")
])
prompt1 = chat_prompt_template.invoke({"name":"小智", "question":"你能帮我做什么"})
print(type(prompt1))   # <class 'langchain_core.prompt_values.ChatPromptValue'>
print(prompt1)

## 1.2 from_message
chat_prompt_template2 = ChatPromptTemplate.from_messages([
    ("system", "你是一个AI助手. 你的名字是{name}."),
    ("human", "我的问题是{question}.")
])
prompt2 = chat_prompt_template2.invoke({"name":"小明", "question":"Ai是什么"})
print(prompt2)


########################################
# 2.调用提示词模版的几种方法: invoke() \ format() \ format_messages() \ format_prompt()
## 2.1 invoke() 传入的是字典, 返回的是 ChatPromptValue  (前面讲过)

## 2.2 format() 传入的是变量名和值, 返回的是 str
prompt3 = chat_prompt_template2.format(name="小明", question="Ai是什么")
print(type(prompt3))
print(prompt3)

## 2.3 format_message() 传入的是变量名和值, 返回的是 list  (推荐使用)
prompt4 = chat_prompt_template2.format_messages(name="小明", question="Ai是什么")
print(type(prompt4))
print(prompt4)

## 2.4 format_prompt() 传入的是变量名和值, 返回的是 ChatPromptValue
prompt5 = chat_prompt_template2.format_prompt(name="小明", question="Ai是什么")
print(type(prompt5))
print(prompt5)


########################################
# 3.更加丰富的实例化参数类型
# 参数是列表类型，列表的元素可以是字符串、字典、字符串构成的元组、消息类型、提示词模板类型、消息提示词模板类型等
## 3.1 字符串类型  （不推荐），因为默认角色都是human
chat_template3 = ChatPromptTemplate.from_messages(
    [
        "Hello, {name}!"  # 等价于 ("human", "Hello, {name})!"
    ]
)
print(chat_template3.format_messages(name="小谷AI"))

## 3.2 dict 类型
chat_template3 = ChatPromptTemplate.from_messages([
    {"role":"system", "content":"你是一个{role}."},
    {"role":"human", "content": ["复杂内容", {"type":"text"}]}
])
print(chat_template3.format_messages(role="教师"))

## 3.3 Message 类型
chat_template3 = ChatPromptTemplate.from_messages([
    SystemMessage(content="我是一个贴心的智能助手"),
    HumanMessage(content="我的问题是:人工智能英文怎么说？")
])
print(chat_template3.format_messages())

## 3.4 BaseChatPromptTemplate 类型
chat_temp1 = ChatPromptTemplate.from_messages([("system","我是一个人工智能助手，我的名字叫{name}")])
chat_temp2 = ChatPromptTemplate.from_messages([("human","很高兴认识你,我的问题是{question}")])
chat_temp3 = ChatPromptTemplate.from_messages([
    chat_temp1,chat_temp2
])
print(chat_temp3.format_messages(name="小智",question="你为什么这么帅？"))

## 3.5 BaseMessagePromptTemplate 类型  (和 3.3 的区别，3.3 的模版入参可以为空, 3.5 不可以)
system_message_prompt = SystemMessagePromptTemplate.from_template("你是一个专家{role}")
human_message_prompt = HumanMessagePromptTemplate.from_template("给我解释{concept}，用浅显易懂的语言")
chat_template5 = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
print(chat_template5.format_messages(role="物理学家", concept="相对论"))


########################################
# 4.结合LLM
chat_model = ChatOpenAI(
    model="gpt-4o-mini"
)
chat_template4 = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
    HumanMessagePromptTemplate.from_template("hello 你好吗？"),
    AIMessagePromptTemplate.from_template("我很好 谢谢!"),
    HumanMessagePromptTemplate.from_template("{query}")
])
# res = chat_model.invoke(chat_template4.format(product="AGI课堂",name="Bob",query="你是谁"))
# print(res.content)


########################################
# 5.插入消息列表: MessagePlaceholder (可以存储对话历史内容)
# 当 ChatPromptTemplate 模版中的消息类型和个数不确定的时候，就可以使用 MessagePlaceholder
# 使用场景：多轮对话系统存储历史消息
chat_template5 = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    MessagesPlaceholder("history"),
    ("human","{question}")
])
prompt5 = chat_template5.format_messages(
    history=[HumanMessage(content="1+2*3 = ?"), AIMessage(content="1+2*3=7")],
    question="我刚才问题是什么？"
)
res = chat_model.invoke(prompt5)
print(res.content)