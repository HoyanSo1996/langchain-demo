from dotenv import load_dotenv
from langchain_core.prompts import load_prompt

load_dotenv()

########################################
# 1. yaml格式提示词
prompt = load_prompt("asset/prompt.yaml", encoding="utf-8")
print(prompt.format(name="年轻人", what="滑稽"))


# 2. json格式提示词
prompt = load_prompt("asset/prompt.json",encoding="utf-8")
print(prompt.format(name="张三",what="搞笑的"))