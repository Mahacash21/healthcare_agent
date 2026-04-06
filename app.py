import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Healthcare Policy Assistant",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Healthcare Policy Assistant")
st.markdown("Ask any question about your healthcare policy document.")
st.divider()

# Load vector DB only once using Streamlit cache
@st.cache_resource
def load_chain():
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory="./db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate.from_template("""
    You are a helpful healthcare insurance assistant.
    Use the following context from the policy document to answer the question.
    If you don't know the answer, say "I could not find this in the policy document."
    Do not make up answers.

    Context: {context}

    Question: {question}

    Answer:""")

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Load the chain
chain = load_chain()

# Chat history — store in session so it persists during the session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input at bottom of page
if question := st.chat_input("Ask a question about your policy..."):

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Get and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching policy..."):
            answer = chain.invoke(question)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})