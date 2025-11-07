import os
import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
# 1. 创建大模型
chat_model = ChatOpenAI(
    model="gpt-4o-mini"
)

# 2. 准备提示词
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("你是{product}的客服助手。你的名字叫{name}"),
    HumanMessagePromptTemplate.from_template("hello 你好吗？"),
    AIMessagePromptTemplate.from_template("我很好 谢谢!"),
    HumanMessagePromptTemplate.from_template("{query}")
]).partial(product="AI", name="小智")

# 3. 准备解析器
parser = StrOutputParser()

# 4. 构建链式调用（LCEL语法，LangChain Expression Language）
chain = prompt_template | chat_model | parser

response = chain.invoke({"query":"解释一下 langchain 是什么"})
print(response)