# LangChain: Summarize Text From URL or PDF


## Overview
This project is a Streamlit-based application that summarizes content from URLs (including YouTube videos) and PDF documents using LangChain and Groq's LLM (gemma2-9b-it). The application extracts text from web pages, YouTube transcripts, or PDF files, then summarizes the content using refined prompts and a summarization chain.


## Features
- Supports summarization of text from:
  - Website URLs
  - YouTube videos
  - Uploaded PDF files
- Uses LangChain's `load_summarize_chain` with `map_reduce` and `refine` techniques
- Provides structured summaries in bullet points
- Secure API key entry for Groq LLM
- Uses `RecursiveCharacterTextSplitter` for chunking long documents


## Tech Stack
- **Python**
- **Streamlit** (for the UI)
- **LangChain** (for LLM interaction)
- **Groq API** (for summarization)
- **UnstructuredURLLoader** (for loading web pages)
- **YoutubeLoader** (for extracting YouTube transcripts)
- **PyPDFLoader** (for PDF processing)


## Usage
1. **Enter your Groq API Key** in the sidebar.
2. **Select the input type:**
   - `URL` (Paste a webpage or YouTube link)
   - `PDF` (Upload a document)
3. **Click "Summarize the Content"** to generate the summary.
4. **View the summarized content** in the application.


## Environment Variables
Set up your Groq API Key as an environment variable:
```bash
export GROQ_API_KEY="your_api_key_here"
```


## Dependencies
The required dependencies are listed in `requirements.txt`. Example:
```text
langchain
langchain-community
langchain-text-splitters
langchain-groq
python-dotenv
youtube_transcript_api
unstructured
validators==0.28.1
pytube
pypdf
pymupdf
streamlit
transformers
```


## Author
**Likith Sagar**