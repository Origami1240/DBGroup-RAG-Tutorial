from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from langchain_community.document_loaders import DirectoryLoader


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# 设置环境变量
PARATERA_API_KEY = "sk-YkAdAuw2r0xdfnkld4Fi4w"
PARATERA_LLM_MODEL = "DeepSeek-R1-0528"
PARATERA_EMBEDDING_MODEL = "GLM-Embedding-3"
PARATERA_BASE_URL = "https://llmapi.paratera.com/v1"

os.environ["OPENAI_API_KEY"] = PARATERA_API_KEY
os.environ["OPENAI_BASE_URL"] = PARATERA_BASE_URL

app = FastAPI()

# 请求和响应模型
class Question(BaseModel):
    query: str

class Answer(BaseModel):
    answer: str
    context: List[str]

# 初始化组件
def init_rag():
    print("[init_rag] 开始加载 testdoc/readme.md ...")
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    loader = UnstructuredMarkdownLoader("testdoc/README.md")
    documents = loader.load()
    print(f"[init_rag] 文档加载完成，数量: {len(documents)}")

    print("[init_rag] 开始文档分割...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    print(f"[init_rag] 文档分割完成，分块数: {len(texts)}")

    print("[init_rag] 初始化 Embeddings...")
    embeddings = OpenAIEmbeddings(
        model=PARATERA_EMBEDDING_MODEL,
        api_key=PARATERA_API_KEY,
        base_url=PARATERA_BASE_URL,
    )
    print("[init_rag] Embeddings 初始化完成")

    print("[init_rag] 创建向量存储...")
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("[init_rag] 向量存储创建完成")

    print("[init_rag] 初始化 LLM...")
    llm = ChatOpenAI(
        model=PARATERA_LLM_MODEL,
        api_key=PARATERA_API_KEY,
        base_url=PARATERA_BASE_URL,
        temperature=0.1,
    )
    print("[init_rag] LLM 初始化完成")

    print("[init_rag] 创建自定义提示模板...")
    prompt_template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

    上下文：
    {context}

    问题：{question}

    回答："""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    print("[init_rag] 提示模板创建完成")

    print("[init_rag] 创建 QA 链...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("[init_rag] QA 链创建完成")

    return qa_chain

# 初始化 RAG
qa_chain = init_rag()

@app.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    try:
        # 获取答案
        result = qa_chain({"query": question.query})
        
        # 提取上下文
        contexts = [doc.page_content for doc in result["source_documents"]]
        
        return Answer(
            answer=result["result"],
            context=contexts
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
