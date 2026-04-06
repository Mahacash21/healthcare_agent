from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Step 1 — Load existing vector DB
print("Loading vector database...")
embeddings = OpenAIEmbeddings()
vectorstore = Chroma(
    persist_directory="./db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("✅ Vector database loaded")

# Step 2 — Custom healthcare prompt
prompt = PromptTemplate.from_template("""
You are a helpful healthcare insurance assistant.
Use the following context from the policy document to answer the question.
If you don't know the answer, say "I could not find this in the policy document."
Do not make up answers.

Context: {context}

Question: {question}

Answer:""")

# Step 3 — Build the chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 4 — Interactive loop
print("\n🏥 Healthcare Policy Assistant Ready!")
print("Type 'exit' to quit\n")

while True:
    question = input("Your question: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if question.strip() == "":
        continue

    print("\nSearching policy...\n")
    answer = chain.invoke(question)
    print(f"Answer: {answer}")
    print("-" * 50 + "\n")