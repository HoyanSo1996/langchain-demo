import os

import dotenv
from langchain_classic.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_BASE_URL")

########################################

# 1. 连接 MySQL 数据库
db_user = "root"
db_password = "root"
db_host = "127.0.0.1"
db_port = "3306"
db_name = "springcloud_db"
# mysql+pymysql://用户名:密码@ip地址:端口号/数据库名
db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
'''
print("哪种数据库：", db.dialect)
print("获取数据表：", db.get_usable_table_names())
res = db.run("SELECT count(*) FROM user;")  # 执行查询
print("查询结果：", res)
'''

# 2. 创建大模型
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. 构建 chains
chain = create_sql_query_chain(llm=llm, db=db)
response = chain.invoke({"question": "一共有多少个用户？", "table_names_to_use": ["user"]}) # 限制使用的表
print(response)