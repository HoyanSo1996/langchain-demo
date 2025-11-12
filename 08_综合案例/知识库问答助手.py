import os

import dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter


class ChatDoc:

    def __init__(self):
        self.doc = None
        self.splitText = []
        # 分割后的文本 split text
        self.template = [
            ("system",
             "你是一个处理文档的秘书,你从不说自己是一个大模型或者AI助手,你会根据下面提供的上下文内容来继续回答问题.\n 上下文内容\n {context} \n"),
            ("human", "你好！"),
            ("ai", "您好，我是尚硅谷秘书"),
            ("human", "{question}"),
        ]
        self.prompt = ChatPromptTemplate.from_messages(self.template)
        dotenv.load_dotenv()

    def get_file(self):
        doc = self.doc
        loaders = {"docx": Docx2txtLoader}
        file_extension = doc.split(".")[-1]
        loader_class = loaders.get(file_extension)
        if loader_class:
            try:
                loader = loader_class(doc)
                text = loader.load()
                return text
            except Exception as e:
                print(f"Error loading {file_extension} files:{e}")
        else:
            print(f"Unsupported file extension: {file_extension}")
            return None

    # 处理文档的函数
    def split_sentences(self):
        full_text = self.get_file()
        if full_text != None:
            text_split = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separator="\n\n",
                length_function=len,
                is_separator_regex=False

            )
            self.splitText = text_split.split_documents(full_text)

    # 向量化与向量存储
    def embedding_and_vectorDB(self):
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        db = Chroma.from_documents(
            documents=self.splitText,
            embedding=embeddings
        )
        return db

    # 初始化LLM
    def init_llm(self):
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY1"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )

    # 提问并找到相关的文本块
    def ask_and_find_files(self, question):
        db = self.embedding_and_vectorDB()
        retriever = db.as_retriever()
        return retriever.invoke(question)

    # 用自然语言和文档聊天
    def chat_with_doc(self, question):
        content = ""
        db_content = self.ask_and_find_files(question)
        for c in db_content:
            content += c.page_content
        print(f"db_content = {db_content}")
        messages = self.prompt.format_messages(context=db_content, question=question)
        print(f"messages = {messages}" )
        llm = self.init_llm()
        return llm.invoke(messages)


if __name__ == "__main__":
    chat_doc = ChatDoc()
    chat_doc.doc = "asset/13-sgg_chat.docx"
    chat_doc.split_sentences()
    response = chat_doc.chat_with_doc("尚硅谷的地址在哪")
    print(response.content)