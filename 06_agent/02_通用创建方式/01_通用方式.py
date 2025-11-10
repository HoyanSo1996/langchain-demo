# @File    : 01_通用方式.py
# @Author  : Kenny So
# @Date    : 2025/11/9 23:41
# @Version : 1.0
import os

import dotenv
from langchain import hub
from langchain.agents import initialize_agent, AgentType, AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

'''
# 通用方式 (推荐)
# (1) 创建 Agent: create_xxx_agent()
# (2) 创建 AgentExecutor: 调用 AgentExecutor() 的构造方法
'''

# 1.初始化搜索工具
search = TavilySearch(max_results=3)

# 2.封装Tool的实例 （本步骤可以考虑省略，直接使用[search]替换[search_tool]。但建议加上
search_tool = Tool(
    name="Search",
    func=search.run,
    description="用于搜索互联网上的信息"
)

# 3.初始化 llm
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

######################################################
############### 使用 Function_call 模式 ###############
# 4. 创建提示词模版
'''
Tip:
    必须声明 agent_scratchpad，它用于存储和传递Agent的思考过程。比如，在调用链式工具时
    （如先搜索天气再推荐行程），agent_scratchpad 保留所有历史步骤，避免上下文丢失。format 
    方法会将 intermediate_steps 转换为特定格式的字符串，并赋值给 agent_scratchpad 变量。
    如果不传递 intermediate_steps 参数，会导致 KeyError: 'intermediate_steps'错误。
'''
prompt = ChatPromptTemplate.from_messages([
    ("system", "您是一位乐于助人的助手，请务必使用 tavily_search_results_json 工具来获取信息。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 5.创建 AgentExecutor
agent = create_tool_calling_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

# 6.创建 AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True
)

# 7.测试
# res = agent_executor.invoke({"input": "今天广州的天气怎么样？"})
# print(res)


#####################################################
################### 使用 ReAct 模式 ###################
# (2) 使用 ReAct 模式
# 4.创建提示词模版
# 4.1 使用 PromptTemplate
'''
提示词中必须带有 {tool_names}, {input}, {agent_scratchpad}
使用 LangChain Hub 中的官方 ReAct 提示模板
'''
prompt2 = PromptTemplate.from_template('''
    Answer the following questions as best you can. You have access to the following tools: {tools}
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    Begin!
    Question: {input}
    Thought:{agent_scratchpad}'
    ''')

# 4.2 直接拉取官方的 ReAct 提示模板, 不用显示声明
prompt3 = hub.pull("hwchase17/react")
# print("prompt3=", prompt3)

# 4.3 使用 ChatPromptTemplate
# 使用 prompt4 模版会报错, 因为使用ReAct模式时，要求 LLM 的响应必须遵循严格的格式（如包含记）。
# 但 LLM 直接返回了自由文本（非结构化），导致解析器无法识别。 @@@@有疑问
prompt4 = ChatPromptTemplate.from_messages([
    ("system", "你是一个人工智能的助手，在用户提出需求以后，必须要调用search_tool进行联网搜索"),
    ("system", prompt3.template),
    ("system", "当前思考：{agent_scratchpad}"),
    ("human", "我的问题是：{input}")  # 必须在提示词模板中提供agent_scratchpad参数。
])

# 5.创建 Agent 对象
agent2 = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt4
)

# 6.创建 AgentExecutor 执行器
agent_executor2 = AgentExecutor(
    agent=agent2,
    tools=[search_tool],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3  # 可选：限制最大迭代次数，防止无限循环
)

# 7.测试
res = agent_executor2.invoke({"input": "今天佛山的天气怎么样？?"})
print(res)
