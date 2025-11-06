import os
import dotenv
from langchain.output_parsers import DatetimeOutputParser
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, XMLOutputParser, \
    CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

chat_model = ChatOpenAI(
    model="gpt-4o-mini"
)

########################################################################
# 1. 字符串解析器 StrOutputParser
# parser = StrOutputParser()
'''
message = [
    SystemMessage("将以下内容从英文翻译成中文"),
    HumanMessage("It's a nice day today")
]
res = chat_model.invoke(message1)
# print(type(res))
# print(res)
print(parser.invoke(res))  # 等价于 res1.content
'''

########################################################################
# 2. JSON解析器 JsonOutputParser
## 方式1: 用户自己通过提示词指明返回Json格式
# parser = JsonOutputParser()
'''
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个靠谱的{role}"),
    ("human", "{question}")
])
prompt = prompt_template.format_messages(
    role="人工智能专家",
    question="人工智能用英文怎么说？问题用q表示，答案用a表示，返回一个JSON格式"
)
res = chat_model.invoke(prompt)
# print(type(res))
# print(res)
print(parser.invoke(res))
'''

# 方式2：借助JsonOutputParser的get_format_instructions()，生成格式说明，指导模型输出 JSON 结构
# print(parser.get_format_instructions())   # Return a JSON object.
'''
prompt_template = PromptTemplate.from_template("回答用户的查询.{format_instructions}\n{query}\n").partial(
    format_instructions=parser.get_format_instructions()   # 相当于给入提示 "Return a JSON object."
)
prompt = prompt_template.invoke("告诉我一个笑话。")
res = chat_model.invoke(prompt)
print(parser.invoke(res))
'''

# 使用链式写法调用 (由于三个对象调的都是 invoke 方法, 所以可以用管道符连在一起)
'''
prompt_template = PromptTemplate.from_template("回答用户的查询.{format_instructions}\n{query}\n").partial(
    format_instructions=parser.get_format_instructions()   # 相当于给入提示 "Return a JSON object."
)
chain = prompt_template | chat_model | parser
res = chain.invoke({"query": "告诉我一个笑话。"})
print(res)
'''

########################################################################
# 3. XML解析器 XMLOutputParser
# parser = XMLOutputParser()
'''
# query = "生成汤姆·汉克斯的简短电影记录"
# res = chat_model.invoke(f"{query}请将影片附在<movie></movie>标签中")
# print(type(res))   # <class 'langchain_core.messages.ai.AIMessage'>
# print(res.content)
# print(parser.invoke(res))

query = "生成周星驰的简短电影记录"
format_instructions = parser.get_format_instructions()
res = chat_model.invoke(f"{query}\n {format_instructions}")
# print(res)
print(parser.invoke(res))
'''

########################################################################
# 4. 列表解析器 CommaSeparatedListOutputParser
# parser = CommaSeparatedListOutputParser()
'''
messages = "大象,猩猩,狮子"
result = parser.parse(messages)
print(type(result))
print(result)
'''


########################################################################
# 5. 日期解析器 DatetimeOutputParser
parser = DatetimeOutputParser()

prompt_template = ChatPromptTemplate.from_messages([
    ("system","{format_instructions}"),
    ("human","{request}")
])
chain = prompt_template | chat_model | parser
res = chain.invoke({
    "request": "中华人民共和国是什么时候成立的",
    "format_instructions": parser.get_format_instructions()
})
print(type(res))
print(res)
