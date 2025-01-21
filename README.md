
---

This project is a modular and interactive Streamlit application that leverages the Retrieval-Augmented Generation (RAG) approach for answering questions based on document content. It integrates LangChain, FAISS, and Athina for document processing, embedding creation, retrieval, and evaluation.

## File Structure

```
streamlit_app/
├── app.py             # Main Streamlit app
├── utils.py           # Helper functions for processing and retrieval
├── sample.pdf         # Example PDF for testing
└── requirements.txt   # Python dependencies
```

---

## Features

- **Document Upload**: Upload PDF documents to extract and process content.
- **RAG Workflow**: Implements a RAG chain for generating answers to user queries.
- **Evaluation**: Evaluates responses using Athina's `DoesResponseAnswerQuery` model.
- **Interactive UI**: Streamlit-powered user interface for ease of use.
- **Modular Codebase**: Utilizes a separate `utils.py` file for helper functions to keep the code clean and organized.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/streamlit-rag-athina
   cd streamlit-rag-athina
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables for your API keys:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export ATHINA_API_KEY=your_athina_api_key
   ```

---

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the address displayed in the terminal (e.g., `http://localhost:8501`).

3. Upload a PDF document via the sidebar.

4. Ask questions about the document, and view the RAG-generated responses.

5. Evaluate the generated responses using Athina's evaluation tools.

---

## Dependencies

The required Python packages are listed in `requirements.txt`:
```plaintext
streamlit
langchain-openai
athina
faiss-gpu
pytesseract
unstructured-client
unstructured[all-docs]
langchain
datasets
pandas
```

Install them with:
```bash
pip install -r requirements.txt
```

---

## Key Modules

### app.py
- Main Streamlit application.
- Handles the UI, file uploads, and integration with the RAG workflow.

### utils.py
Contains helper functions for:
- **PDF Parsing**: Extracts content using Unstructured.
- **Document Chunking**: Converts parsed content into LangChain documents.
- **Embeddings**: Generates embeddings using OpenAI.
- **Vectorstore**: Creates and retrieves from FAISS vectorstores.
- **RAG Chain**: Constructs the RAG workflow.

---

## How It Works

1. **PDF Parsing**:
   - The uploaded PDF is processed using `unstructured` to extract and categorize content.

2. **LangChain Integration**:
   - The parsed elements are converted into LangChain documents.
   - OpenAI embeddings are generated for the documents.

3. **Vectorstore and Retrieval**:
   - FAISS is used to create a vectorstore, enabling fast and efficient retrieval.

4. **RAG Workflow**:
   - A RAG chain is created with a ChatOpenAI model for answering user queries.

5. **Athina Evaluation**:
   - The responses are evaluated against the input questions for accuracy using Athina's evaluation tools.

---

## Testing

A sample PDF (`sample.pdf`) is included in the repository for testing the application. Place your PDF documents in the `streamlit_app` directory to use them with the app.

---

## Future Enhancements

- Support for additional document formats.
- Advanced evaluation metrics and models.
- Deployment on cloud platforms like AWS, Azure, or Streamlit Cloud.

---

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

Let me know if you need further assistance with any part of this project! 🚀
