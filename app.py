import os
import streamlit as st
import google.generativeai as genai

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)

from langchain_community.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate


# ==================================================
# LOAD ENV VARIABLES
# ==================================================
load_dotenv()

genai.configure(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# ==================================================
# EXTRACT TEXT FROM PDF
# ==================================================
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:
            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        except Exception as e:
            st.error(f"Error Reading PDF: {e}")

    return text


# ==================================================
# SPLIT TEXT INTO CHUNKS
# ==================================================
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ==================================================
# CREATE VECTOR STORE
# ==================================================
def get_vector_store(text_chunks):

    if not text_chunks:
        st.error("No text chunks found.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="embedding-004"  
            )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ==================================================
# CREATE CONVERSATIONAL CHAIN
# ==================================================
def get_conversational_chain():

    prompt_template = """
    Answer the question as detailed as possible
    from the provided context.

    If the answer is not available in the context,
    say:
    "Answer is not available in the context."

    Do not provide wrong answers.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        llm=model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


# ==================================================
# HANDLE USER QUESTION
# ==================================================
def user_input(user_question):

    embeddings = GoogleGenerativeAIEmbeddings(
model="embedding-004"    )

    # Load FAISS Database
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Similarity Search
    docs = new_db.similarity_search(user_question)

    # Load QA Chain
    chain = get_conversational_chain()

    # Generate Response
    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )

    st.write("## Reply")
    st.write(response["output_text"])


# ==================================================
# MAIN FUNCTION
# ==================================================
def main():

    st.set_page_config(
        page_title="Chat With Multiple PDFs",
        page_icon="📚",
        layout="wide"
    )

    st.header("📚 Chat With Multiple PDFs Using Gemini AI")

    # User Question
    user_question = st.text_input(
        "Ask a Question From the Uploaded PDFs"
    )

    if user_question:

        if os.path.exists("faiss_index"):

            user_input(user_question)

        else:
            st.warning(
                "Please upload and process PDF files first."
            )

    # SIDEBAR
    with st.sidebar:

        st.title("📂 Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):

            if pdf_docs:

                with st.spinner("Processing PDFs..."):

                    # Extract Text
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.error(
                            "No readable text found in PDF."
                        )
                        return

                    # Split Into Chunks
                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                        st.error(
                            "Text chunking failed."
                        )
                        return

                    # Create Vector Store
                    get_vector_store(text_chunks)

                    st.success(
                        "PDF Processing Completed Successfully!"
                    )

            else:
                st.warning(
                    "Please upload at least one PDF file."
                )


# ==================================================
# RUN APP
# ==================================================
if __name__ == "__main__":
    main()