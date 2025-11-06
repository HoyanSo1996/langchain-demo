# @File    : 06_提示词模版1_PromptTemplate.py
# @Author  : Kenny So
# @Date    : 2025/11/5 20:42
# @Version : 1.0
import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")


########################################
# 1. 实例化提示词模版 (2种方式)
# 方式1: 使用构造方法
# 定义模板, 参数中必须要指明: template、input_variables
prompt_template = PromptTemplate(
    template="你是一个{role}, 你的名字叫{name}.",
    input_variable=["role", "name"]
)
# print(prompt_template)
prompt = prompt_template.format(role="人工智能专家", name="小智")
print(prompt)


# 方式2: 调用from_template()  （推荐, 省了 input_variables 不用写）
prompt_template2 = PromptTemplate.from_template(
    "请给我一个关于{topic}的{type}解释。"
)
# print(prompt_template2)
prompt2 = prompt_template2.format(type="详细", topic="量子力学")
print(prompt2)


########################################
# 2. 两种新的结构形式
# 2.1 部分提示词模版
# 2.1.1 方式1：实例化过程中使用 partial_variables 变量
prompt_template3 = PromptTemplate(
    template="{foo} {bar}!",
    input_variables=["foo","bar"],
    # partial_variables={"foo":"hello", "bar":"world"}
    partial_variables={"foo":"hello"}
)
# prompt3 = template2.format()
prompt3 = prompt_template3.format(bar="world")
print(prompt3)

# 2.1.2 方式2：使用 PromptTemplate.partial() 方法创建部分提示模板
prompt_template4 = PromptTemplate(
    template="{foo} {bar}!",
    input_variables=["foo", "bar"]
 ).partial(foo="hello")
prompt4 = prompt_template4.format(bar="world")
print(prompt4)

full_template = """你是一个{role}，请用{style}风格回答：
问题：{question}"""
# 预填充角色和风格
prompt_template5 = PromptTemplate.from_template(full_template).partial(
    role="资深厨师",
    style="专业但幽默"
)
# 只需提供剩余变量
print(prompt_template5.format(question="如何煎牛排？"))


# 2.2 组合提示词模版
prompt_template6 = (
    PromptTemplate.from_template("Tell me a joke about {topic}")
    + ", make it funny"
    + " and in {language}."
 )
prompt6 = prompt_template6.format(topic="sports", language="spanish")
print(prompt6)


########################################
# 3. 给变量赋值的两种方式： format() / invoke()
# format() 上面已讲过，现在讲第二种
prompt_template7 = PromptTemplate.from_template(
    "Tell me a {adjective} joke about {content}."
)
# 3.默认使用f-string进行格式化（返回格式好的字符串）
prompt7 = prompt_template7.invoke({
    "adjective": "funny", "content": "chickens"
})
print(type(prompt7))
print(prompt7)
print("####################################")


########################################
# 4. 结合大模型的使用
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

prompt_template8 = PromptTemplate.from_template(
    "请评价{product}的优缺点，包括{aspect1}和{aspect2}。"
)
prompt8 = prompt_template8.invoke({"product": "联想电脑", "aspect1": "性能", "aspect2": "电池"})
print(llm.invoke(prompt8).content)