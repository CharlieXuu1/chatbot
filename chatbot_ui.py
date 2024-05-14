import streamlit as st
import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

# Load environment variables
dotenv.load_dotenv('key.env')
openai_api_key = os.getenv("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")

# Initialize the OpenAI chat model
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2, openai_api_key="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")

# Load and split documents
loader = PyPDFDirectoryLoader("BidgelyStarterContent2024-02-14")
data = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create embeddings and vector store
openai_embeddings = OpenAIEmbeddings(openai_api_key="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")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=openai_embeddings)

retriever = vectorstore.as_retriever(k=4)

# Define the system template
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context.
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# Streamlit UI
st.title("Bidgely Assistant")
st.write("I am the Bidgely assistant. How can I help you today?")

question = st.text_input("Your question:")
if st.button("Ask"):
    if question:
        context_docs = retriever.invoke(question)
        response = document_chain.invoke(
            {
                "context": context_docs,
                "messages": [HumanMessage(content=question)],
            }
        )
        st.write(f"**Bidgely Assistant:** {response}")

if st.button("Exit"):
    st.write("Exiting the assistant. Goodbye!")
    st.stop()

# Example questions
st.write("Example questions:")
st.write("- How do you detect an EV on the grid?")
st.write("- Tell me more about charger type detection.")
st.write("- What sets Bidgely's EV intelligence apart?")
