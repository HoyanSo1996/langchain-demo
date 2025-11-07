import json
import os

import dotenv
from langchain_community.tools import MoveFileTool
from langchain_core.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################
# 1.定义LLM模型
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# 2.定义工具
tools = [MoveFileTool()]

# 3.这里需要将工具转换为openai函数，后续再将函数传入模型调用
#   因为 OpenAi 大模型 invoke 调用时, 需要传入函数的列表, 所以需要将工具转换为函数: convert_to_openai_function()
functions = [convert_to_openai_function(t) for t in tools]
# print(functions[0])

# 4.提供大模型调用的消息列表
messages = [HumanMessage(content="将本目录下的abc.txt文件移动到C:\\Users\\LPS\\Desktop\\cc")]     # 如果发有关的问题, 返回的 content 为空
# messages = [HumanMessage(content="今天的天气怎么样？")]   # 如果发无关的问题, 返回的 content 会有拒绝内容

# 5.模型使用函数
response = chat_model.invoke(
    input=messages,
    functions=functions
)
print(response)

# 6. 检查是否需要调用工具
if "function_call" in response.additional_kwargs:
    tool_name = response.additional_kwargs["function_call"]["name"]
    tool_args = json.loads(response.additional_kwargs["function_call"]["arguments"])
    print(f"调用工具: {tool_name}, 参数: {tool_args}")
    # 7. 实际执行工具调用
    if "move_file" in response.additional_kwargs["function_call"]["name"]:
        tool = MoveFileTool()
        result = tool.run(tool_args)
        # 执行工具
        print("工具执行结果:", result)
else:
    print("模型回复:", response.content)
