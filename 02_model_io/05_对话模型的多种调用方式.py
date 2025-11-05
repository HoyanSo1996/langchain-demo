import asyncio
import os
import time

import dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 加载配置文件
dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL")

####################################
# 1.大模型流式响应
"""
# 初始化大模型
chat_model = ChatOpenAI(
    model="gpt-4o-mini",
    streaming=True  # 启用流式输出
)

# 创建消息
messages = [HumanMessage(content="你好，请介绍一下自己")]

# 流式调用 LLM 获取响应
print("流式输出开始 =======")
for chunk in chat_model.stream(messages):
    # 逐个打印内容块
    # end = "" 表示无换行符; flush=True 表示刷新缓冲区(如果缓冲区未刷新，内容可能不会立即显示)
    print(chunk.content, end="", flush=True)
print("\n流式输出结束 =======")
"""


####################################
# 2. 大模型批量调用
"""
# 初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")

# 准备消息
messages1 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是机器学习")
]
messages2 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是AIGC")
]
messages3 = [
    SystemMessage(content="你是一位乐于助人的智能小助手"),
    HumanMessage(content="请帮我介绍一下什么是大模型技术")
]
messages = [messages1, messages2, messages3]

# 调用batch
response = chat_model.batch(messages)
print(response)
"""


####################################
# 3. 大模型同步与异步调用
# 初始化大模型
chat_model = ChatOpenAI(model="gpt-4o-mini")

# 同步调用（对比组）
def sync_test():
    messages1 = [
        SystemMessage(content="你是一位乐于助人的智能小助手"),
        HumanMessage(content="请帮我介绍一下什么是机器学习")
    ]
    start_time = time.time()
    response = chat_model.invoke(messages1)
    # 同步调用
    duration = time.time() - start_time
    print(f"同步调用耗时：{duration:.2f}秒")
    return response, duration

# 异步调用（实验组）
async def async_test():
    messages1 = [
        SystemMessage(content="你是一位乐于助人的智能小助手"),
        HumanMessage(content="请帮我介绍一下什么是机器学习")
    ]
    start_time = time.time()
    response = await chat_model.ainvoke(messages1)
    # 异步调用
    duration = time.time() - start_time
    print(f"异步调用耗时：{duration:.2f}秒")
    return response, duration


# 运行测试
if __name__ =="__main__":
    # 运行同步测试
    sync_response, sync_duration = sync_test()
    print(f"同步响应内容: {sync_response.content[:100]}...\n")

    # 运行异步测试
    async_response, async_duration = asyncio.run(async_test())
    print(f"异步响应内容: {async_response.content[:100]}...\n")

    # 并发测试 - 修复版本
    print("\n=== 并发测试 ===")
    start_time = time.time()


    async def run_concurrent_tests():
        # 创建3个异步任务
        tasks = [async_test() for _ in range(3)]
        # 并发执行所有任务
        return await asyncio.gather(*tasks)

    # 执行并发测试
    results = asyncio.run(run_concurrent_tests())

    total_time = time.time() - start_time
    print(f"\n3个并发异步调用总耗时: {total_time:.2f}秒")
    print(f"平均每个调用耗时: {total_time / 3:.2f}秒")