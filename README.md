<div align="center">

# 🎬 VideoMind AI

### 🤖 AI-Powered YouTube Video Assistant using LangChain & RAG

<p>
  <strong>Ask questions about YouTube videos and get intelligent, context-grounded answers directly from the video's transcript.</strong>
</p>

<br>

<img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white">
<img src="https://img.shields.io/badge/FAISS-Vector%20Search-0467DF?style=for-the-badge">
<img src="https://img.shields.io/badge/HuggingFace-LLM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black">
<img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

<br><br>

</div>

---

## 🌟 Overview

**VideoMind AI** is an AI-powered YouTube video question-answering application built using **LangChain, Retrieval-Augmented Generation (RAG), FAISS, Hugging Face and Streamlit**.

Instead of manually watching an entire video to find specific information, users can provide a YouTube URL and interact with the video's content using natural language.

The application:

<ol>
<li>Extracts the video's transcript</li>
<li>Splits the transcript into meaningful chunks</li>
<li>Converts the chunks into vector embeddings</li>
<li>Stores the embeddings inside a FAISS vector database</li>
<li>Retrieves the most relevant information for a question</li>
<li>Passes the retrieved context to an LLM</li>
<li>Generates a context-aware answer</li>
</ol>

<br>

<div align="center">

### 💡 <i>"Turn hours of video content into an interactive AI conversation."</i>

</div>

---

# 🚀 Key Features

<table>
<tr>
<td width="50%">

### 🎥 YouTube Transcript Processing

Automatically extracts transcript content from a YouTube video using the YouTube Transcript API.

</td>

<td width="50%">

### 🧠 Retrieval-Augmented Generation

Uses RAG to retrieve relevant information before generating an answer.

</td>
</tr>

<tr>
<td>

### 🔎 Semantic Search

FAISS performs efficient similarity search over transcript embeddings.

</td>

<td>

### 🤖 LLM-Powered Answers

Uses Hugging Face models to generate natural-language responses.

</td>
</tr>

<tr>
<td>

### 🔗 LangChain LCEL

Uses LangChain Runnable components to build a clean retrieval-generation pipeline.

</td>

<td>

### 💬 Interactive Chat

Ask multiple questions about the analyzed video through a conversational interface.

</td>
</tr>

<tr>
<td>

### 📊 Video Analytics

Displays useful information such as transcript status and processed chunks.

</td>

<td>

### 🎨 Streamlit Interface

Clean and professional web interface built with Streamlit.

</td>
</tr>
</table>

---

# 🏗️ System Architecture

<div align="center">

```text
                    ┌─────────────────────┐
                    │    YouTube URL      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Transcript API      │
                    │                     │
                    │ Extract Transcript  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Text Splitter       │
                    │                     │
                    │ Chunking + Overlap  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hugging Face        │
                    │ Embeddings           │
                    │                     │
                    │ 384-D Vectors       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ FAISS Vector Store  │
                    │                     │
                    │ Similarity Search   │
                    └──────────┬──────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Retriever           │
                    │                     │
                    │ Top-K Relevant      │
                    │ Documents           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prompt Template     │
                    │                     │
                    │ Context + Question  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Hugging Face LLM    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ AI Generated Answer │
                    └─────────────────────┘
