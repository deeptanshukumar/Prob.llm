# 📚 Prob.llm - AI-Powered Study Assistant

A powerful RAG (Retrieval Augmented Generation) study assistant that combines **BM25** and **Semantic Search** with **Reciprocal Rank Fusion (RRF)** for intelligent document analysis and question answering.

![Main Interface](Assets/mainview.png)

## 🌟 Features

### 🔍 **Advanced Hybrid Retrieval System**
- **BM25 Retrieval**: Keyword-based search for exact term matching
- **Semantic Search**: Vector-based similarity search using embeddings
- **Reciprocal Rank Fusion (RRF)**: Combines both methods with equal weights (0.5/0.5)
- **Optimized Performance**: Retrieves 10 documents from each method for faster processing

### 📄 **Multi-Format Document Support**
- **PDF**: Using PyMuPDF for accurate text extraction
- **DOCX**: Microsoft Word document processing
- **PPTX**: PowerPoint presentation support
- **Images**: JPG, JPEG, PNG with OCR capabilities
- **Text**: Plain text file support

### 🤖 **LLM Integration**
- **Ollama Integration**: Local LLM deployment with `llama3.2:1b`
- **Nomic Embeddings**: High-quality text embeddings with `nomic-embed-text`
- **Custom Prompting**: Structured, point-wise answer formatting

### 🖥️ **User-Friendly Interface**
- **Streamlit Web App**: Clean, intuitive interface
- **Real-time Chat**: Interactive Q&A with your documents
- **Document Management**: Easy upload and ingestion
- **Source Citations**: View source documents for answers

![Sources View](Assets/sources.png)

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Ollama** installed and running ([Download Ollama](https://ollama.ai/))

![Ollama Setup](Assets/ollama.png)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/deeptanshukumar/Prob.llm.git
   cd Prob.llm
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama models**
   ```bash
   # Install the LLM model
   ollama pull llama3.2:1b
   
   # Install the embedding model
   ollama pull nomic-embed-text
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📖 Usage

### 1. **Document Upload**
- Place your study materials in the `uploaded_docs/` folder, or
- Use the web interface to upload files directly
- Supported formats: PDF, DOCX, PPTX, JPG, PNG, JPEG, TXT

### 2. **Document Processing**
- The system automatically detects and processes uploaded documents
- Documents are chunked (1000 characters with 200 overlap) for optimal retrieval
- Both BM25 and semantic indexes are created

### 3. **Ask Questions**
- Type your questions in the chat interface
- The system uses hybrid retrieval to find relevant information
- Answers include source document references

### 4. **View Sources**
- Click on source documents to see the original content
- Verify information accuracy and explore related topics

## 🏗️ Architecture

### **Modular Design**
```
prob_llm_assistant/
├── core/                    # Core functionality modules
│   ├── document_loaders.py  # Multi-format document loading
│   ├── text_processing.py   # Chunking and preprocessing
│   ├── vector_store.py      # ChromaDB vector operations
│   ├── retrieval.py         # Hybrid retrieval (BM25 + Semantic + RRF)
│   └── llm_interface.py     # Ollama LLM integration
├── streamlit_app.py         # Web interface
├── main.py                  # Core orchestration logic
└── requirements.txt         # Dependencies
```

### **Hybrid Retrieval Pipeline**
1. **Document Ingestion**: Multi-format parsing and text extraction
2. **Text Processing**: Intelligent chunking with overlap
3. **Dual Indexing**: 
   - BM25 index for keyword search
   - Vector embeddings for semantic search
4. **Query Processing**: 
   - Retrieve 10 documents via BM25
   - Retrieve 10 documents via semantic search
   - Apply RRF to combine and rank results
5. **LLM Generation**: Context-aware answer generation

![System Architecture](Assets/llama.jpg)

## ⚙️ Configuration

### **Model Configuration** (in `main.py`)
```python
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # Ollama embedding model
LLM_MODEL_NAME = "llama3.2:1b"             # Ollama LLM model
```

### **Retrieval Parameters**
```python
# Chunking settings
chunk_size = 1000      # Characters per chunk
chunk_overlap = 200    # Overlap between chunks

# Retrieval settings
k = 10                 # Documents retrieved per method
bm25_weight = 0.5      # BM25 importance weight
semantic_weight = 0.5  # Semantic search importance weight
```

### **Performance Optimization**
- **Reduced retrieval count**: 10 docs per method (down from 15)
- **Optimized chunking**: 1000 characters with 200 overlap
- **Caching**: Streamlit caching for repeated operations
- **Lightweight model**: Using `llama3.2:1b` for speed

## 🔧 Advanced Usage

### **Custom Prompting**
Edit `prompt.txt` to customize the AI's response style:
```
You are an expert assistant. Your goal is to provide a comprehensive and accurate answer based *only* on the provided context.
Synthesize the information from all relevant document chunks to form a cohesive answer.
Do not add any information that is not present in the context.
If the context does not contain the answer, state that clearly.
Try to give the answers in a proper structured format point wise.

Context:
{context}

Question: {question}
```

### **Different LLM Models**
For better performance, you can use different models:
```bash
# Faster, smaller models
ollama pull phi3.5:3.8b-mini-instruct-q4_0
ollama pull gemma2:2b-instruct-q4_0

# More capable models
ollama pull llama3.1:8b
ollama pull mistral:7b
```

### **Batch Document Processing**
Place multiple documents in `uploaded_docs/` and restart the app for batch processing.

## 🛠️ Development

### **Project Structure**
- **`core/`**: Modular components for easy maintenance
- **`streamlit_app.py`**: UI layer separated from business logic
- **`main.py`**: Orchestration and caching logic
- **`uploaded_docs/`**: Document storage directory
- **`chroma_data/`**: Vector database storage

### **Testing**
```bash
# Test hybrid retrieval system
python test_hybrid_retrieval.py

# Test individual components
python -c "from core.retrieval import get_hybrid_retriever; print('Retrieval system ready!')"
```

### **Contributing**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📊 Performance

### **Typical Response Times**
- **Document Loading**: 2-5 seconds (one-time)
- **First Query**: 3-8 seconds (model loading)
- **Subsequent Queries**: 1-2 seconds
- **Document Ingestion**: ~1 second per page

### **System Requirements**
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB for models + document storage
- **CPU**: Modern multi-core processor recommended
- **GPU**: Optional, but improves performance if supported by Ollama

## ❓ Troubleshooting

### **Common Issues**

1. **"Module not found" errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama connection errors**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Verify models are installed
   ollama list
   ```

3. **PDF loading errors**
   ```bash
   pip install pymupdf
   ```

4. **Performance issues**
   - Try smaller models: `phi3.5:3.8b-mini-instruct-q4_0`
   - Reduce chunk_size to 800
   - Use semantic-only retrieval for maximum speed

### **Error Logs**
Check the Streamlit interface for detailed error messages and suggestions.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: Framework for LLM applications
- **Ollama**: Local LLM deployment platform
- **ChromaDB**: Vector database for embeddings
- **Streamlit**: Web application framework
- **PyMuPDF**: PDF processing library

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/deeptanshukumar/Prob.llm/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deeptanshukumar/Prob.llm/discussions)
- **Documentation**: Check this README for comprehensive setup and usage instructions

---

**Made with ❤️ for students and researchers who want AI-powered document analysis**