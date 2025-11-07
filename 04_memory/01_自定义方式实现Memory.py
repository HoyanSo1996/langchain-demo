import os

import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
# 创建大模型实例
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_with_model(question):
    # 步骤一：初始化消息
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是一位人工智能小助手"),
        ( "human", "{question}")
    ])

    # 步骤二：定义一个循环体：
    while True:
        # 步骤三：调用模型
        chain = chat_prompt_template | llm
        response = chain.invoke({"question": question})
        # 步骤四：获取模型回答
        print(f"模型回答: {response.content}")
        # 询问用户是否还有其他问题
        user_input = input("您还有其他问题想问嘛？(输入'退出'结束对话)")
        # 设置结束循环的条件
        if (user_input == "退出"):
            break

        # 步骤五：记录用户回答
        chat_prompt_template.messages.append(AIMessage(content=response.content))
        chat_prompt_template.messages.append(HumanMessage(content=user_input))


chat_with_model("你好")
