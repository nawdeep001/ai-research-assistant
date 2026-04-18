import streamlit as st
from openai import OpenAI

from utils.pdf_loader import extract_text
from utils.embeddings import chunk_text, get_embeddings
from utils.retriever import VectorStore

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"],
    base_url=st.secrets["OPENAI_BASE_URL"]
)

# Cache function to process PDF and store results
@st.cache_data
def process_pdf(file):
    text = extract_text(file)
    chunks = chunk_text(text)
    embeddings = get_embeddings(chunks)
    return chunks, embeddings

st.title("📄 AI Research Assistant (RAG)")


#UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file:
#     text = extract_text(uploaded_file)

#     st.success("Processing PDF...")

#     # STEP 1: Chunking
#     chunks = chunk_text(text)

#     # STEP 2: Embeddings
#     embeddings = get_embeddings(chunks)

#     # STEP 3: Store in FAISS
#     vector_store = VectorStore(embeddings, chunks)

#     st.success("Ready! Ask questions 👇")

#     query = st.text_input("Ask a question:")

#     if query:
#         # Query embedding
#         query_embedding = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=query
#         ).data[0].embedding

#         # Retrieve relevant chunks
#         relevant_chunks = vector_store.search(query_embedding)

#         context = "\n\n".join(relevant_chunks)

#         # GPT call
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "Answer using only the given context."},
#                 {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
#             ]
#         )

#         st.write(response.choices[0].message.content)

if uploaded_file:
    chunks, embeddings = process_pdf(uploaded_file)
    vector_store = VectorStore(embeddings, chunks)

    st.success("PDF ready! Ask questions below 👇")

    user_query = st.chat_input("Ask something about the document")

    if user_query:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("user"):
            st.write(user_query)

        # Get query embedding
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query
        ).data[0].embedding

        # Retrieve chunks
        relevant_chunks = vector_store.search(query_embedding)

        context = "\n\n".join(relevant_chunks)

        # GPT response
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "Answer using only the context."},
        #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        #     ]
        # )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a research assistant. Answer clearly and cite relevant parts from context."
                },
                {
                     "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {user_query}"
                }
            ]
        )

        answer = response.choices[0].message.content

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.write(answer)




        with st.expander("📌 Sources used"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"**Chunk {i+1}:**")
                st.write(chunk[:300])
                st.write("---")