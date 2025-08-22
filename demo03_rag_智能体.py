"""
RAG - 检索增强生成示例
@File       demo03_rag_智能体.py
@Author     小明
@Date       2025/8/22 10:29
@Version    V0.0.1
"""
import streamlit as st
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_embedding, get_model

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True, # 返回问答的 HumanMessage 和 AIMessage
        memory_key='chat_history', # 由于使用 ConversationalRetrievalChain 对话链返回的字典中使用到的是 'chat_history' 保存
    )


def qa_agent(_file, _question):
    """
    问答智能体功能函数
    :param _file: 选择加载的 pdf 文件
    :param _question: 针对装载pdf文件的提问问题
    :return: 针对装载pdf文件的问题答案
    """
    # 将选择上传的 pdf 文件先暂存到一个临时文件中
    file_path = 'temp.pdf'
    with open(file_path, 'wb') as f:
        f.write(_file.read())
    # 装载文档, 将保存在临时文件中的文档内容装载出来（相当于加载了上传组件选择的pdf文件）
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # 文档分割
    splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '。', '！', '？', '，', '、', ''],
        chunk_size=500,
        chunk_overlap=50,
    )
    chunk_docs = splitter.split_documents(documents)
    # 嵌入(文本向量化)
    embedding = get_embedding()
    # 向量存储
    db = FAISS.from_documents(
        documents=chunk_docs,
        embedding=embedding,
    )
    # 检索器
    retriever = db.as_retriever()
    # 对话链
    llm = get_model()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=st.session_state.memory, # 添加记忆功能，可将对话历史保存起来
    )
    # 提交
    res = chain.invoke({
        'chat_history': [],
        'question': _question,
    })
    print(f'{res = }')
    # 将返回数据中保存在 chat_history 中的对话历史暂存到 session_state 中
    # 以便在页面中显示出各次对话历史数据。
    st.session_state.chat_history = res['chat_history']
    return res['answer']

st.title('千锋PDF智能AI问答系统')

file = st.file_uploader('选择文件:', type='pdf')
question = st.text_area('提问:', placeholder='请针对选择的文件进行提问')

# 选择了文件与填写了问题，则接入大语言模型接口实现问答
if file and question:
    with st.spinner('思考中...'):
        answer = qa_agent(file, question)
    st.markdown('### 答案:')
    st.write(answer)

# 如果有对话历史，则显示出对话历史信息
if 'chat_history' in st.session_state:
    with st.expander('对话历史'):
        # 在对话历史列表中，每两个元素一对作为对话的问和答
        for index in range(0, len(st.session_state.chat_history), 2):
            st.write(f'问题: {st.session_state.chat_history[index].content}')
            st.write(f'答案: {st.session_state.chat_history[index+1].content}')
            st.divider() # 一条分割线
