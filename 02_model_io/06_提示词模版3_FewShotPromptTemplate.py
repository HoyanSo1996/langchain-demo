# @File    : 06_提示词模版3_FewShotPromptTemplate.py
# @Author  : Kenny So
# @Date    : 2025/11/5 20:42
# @Version : 1.0
import os
import dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
chat_model = ChatOpenAI(
    model="gpt-4o-mini"
)

########################################
# 1. 创建示例集合
examples = [
    {"input":"北京天气怎么样","output":"北京市"},
    {"input":"南京下雨吗","output":"南京市"},
    {"input":"武汉热吗","output":"武汉市"}
]

# 2. 创建PromptTemplate实例
example_prompt = PromptTemplate.from_template(
    template="Input: {input}\nOutput: {output}"
)

# 3. 创建FewShotPromptTemplate实例
prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,  # 这里是 示例 的模板
    suffix="Input: {input}\nOutput:",  # 这里是 要求ai输出 的模板。
    input_variables=["input"]  # 传入的变量
)

# 4. 调用
prompt = prompt.invoke({"input":"长沙多少度"})
print("Prompt: " + prompt.to_string())
res = chat_model.invoke(prompt)
print("response: " + res.content)