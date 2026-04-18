# AI Research Assistant (RAG-based)

A simple and practical application that allows users to upload a PDF and ask questions about its content.  
The system retrieves relevant sections from the document and generates answers using GPT.

---

## What it does

- Upload a research paper or any PDF  
- Ask questions in natural language  
- Retrieves relevant parts of the document  
- Generates answers grounded in the document context  
- Displays source chunks used for answering  

---

##  How it works

This project follows a basic **Retrieval-Augmented Generation (RAG)** pipeline:

1. Extract text from the uploaded PDF  
2. Split the text into smaller chunks  
3. Convert chunks into embeddings  
4. Store embeddings using FAISS  
5. On user query:
   - Retrieve the most relevant chunks  
   - Pass them to the language model  
   - Generate a contextual answer  

---

##  Tech Stack

- Python  
- Streamlit  
- OpenAI API (gpt-4o-mini, embeddings)  
- FAISS (vector search)  
- PyPDF2  

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-research-assistant.git
cd ai-research-assistant
````

Create a virtual environment:

```bash
python3 -m venv genai
source genai/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your API key:

Create a file:

```
.streamlit/secrets.toml
```

Add:

```toml
OPENAI_API_KEY = "your_key"
OPENAI_BASE_URL = "your_base_url"
```

Run the app:

```bash
streamlit run app.py
```

---

## Live Demo

Link: https://ai-research-assistant-nawdeep-mnit.streamlit.app

---

## Notes

* Works best for small to medium-sized PDFs
* Large documents may take longer to process
* Embeddings are cached to improve performance

---

## Future Improvements

* Support for multiple PDFs
* Improved chunking strategy
* Enhanced UI for document navigation
* Domain-specific extensions (e.g., agriculture insights)

---

## Author

**Nawdeep Kumar**
M.Tech. Computer Science and Engineering | Email: 283nawdeep@gmail.com

```
