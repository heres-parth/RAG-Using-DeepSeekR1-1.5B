import PyPDF2

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_chunks):
    return embedder.encode(text_chunks)

import faiss
import numpy as np

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_context(query, index, text_chunks, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, k)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return " ".join(relevant_chunks)


import ollama

#def generate_answer(query, context):
#     prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
#    response = ollama.generate(model="deepseek-r1:1.5b", prompt=prompt)
#    return response["response

def generate_answer(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = ollama.generate(model="deepseek-r1:1.5b", prompt=prompt)
    # Clean the output to remove unwanted tags like <think>...</think>
    answer = response["response"].replace("<think>", "").replace("</think>", "").strip()
    return answer


import streamlit as st

DEEPSEEK_LOGO = "deepseek_logo.png"

# Streamlit app
st.title("Chat with Your PDFs using")
st.image(DEEPSEEK_LOGO, width=200)

# Streamlit app
#st.title("Chat with Your PDFs using Deepseek-r1:1.5b")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    # Extract text
    text = extract_text_from_pdf(uploaded_file)
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]  # Split text into chunks

    # Generate embeddings and create FAISS index
    embeddings = generate_embeddings(text_chunks)
    index = create_faiss_index(embeddings)

    # Input question
    query = st.text_input("Ask a question about the PDF:")

    if query:
        # Retrieve relevant context
        context = retrieve_relevant_context(query, index, text_chunks)

        # Generate answer
        answer = generate_answer(query, context)
        st.write("Answer:", answer)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 16px;
    }
    .footer a {
        color: #00B4D8;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        Created by <a href="https://www.linkedin.com/in/parthaland" target="_blank">Parth Aland</a>
    </div>
    """,
    unsafe_allow_html=True
)