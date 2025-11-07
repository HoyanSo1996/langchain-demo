from langchain_core.tools import tool, StructuredTool
from pydantic import BaseModel, Field

# 自定义工具
## 1. 方式一: 使用 @tool 装饰器
class FieldInfo(BaseModel):
    a:int = Field(description="第1个参数")
    b:int = Field(description="第2个参数")

@tool
# @tool(name_or_callable="calculator", description="两个整数相加++", args_schema=FieldInfo, return_direct=True)
def add_number(a:int, b:int) -> int:
    """两个整数相加"""
    return a + b

print(f"name = {add_number.name}")  # 默认是函数名, 可以通过 name_or_callable 重置
print(f"args = {add_number.args}")  # 默认入参信息, 可以通过 args_schema 重置
print(f"description = {add_number.description}")  # 默认是函数内的 """注释""" 内容, 可以通过 description 重置
# 当 return_direct 为True时, 在调用给定工具后, Agent将停止并将结果直接返回给用户。否则直接将结果返回给用户，并不调用工具
print(f"return_direct = {add_number.return_direct}")  # 默认是 False, 可以通过 return_direct 重置
res = add_number.invoke({"a":10,"b" :20})
print(res)
print("################################################")


#####################################################
## 2. 方式二: 使用 StructuredTool.from_function 类方法
#            比 @tool 装饰器更多的可配置性，而无需太多额外的代码。
def search_function(query:str) -> str:
    return "LangChain"

class FieldInfo2(BaseModel):
    query:str = Field(description="要检索的关键词")

search1 = StructuredTool.from_function(
    func=search_function,
    name="Search",
    args_schema=FieldInfo2,
    description="useful for when you need to answer questions about current events",
    return_direct=True
)

print(f"name = {search1.name}")
print(f"description = {search1.description}")
print(f"args = {search1.args}")
print(f"args = {search1.return_direct}")

res = search1.invoke("hello")
print(res)