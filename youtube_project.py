import os
import re
import streamlit as st

from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFaceEndpointEmbeddings
)

from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VideoMind AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background-color: #0b0f19;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background-color: #080c14;
        border-right: 1px solid #20283a;
    }

    /* ---------- HEADINGS ---------- */

    h1 {
        color: #f8fafc !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #f1f5f9 !important;
    }

    /* ---------- TEXT ---------- */

    p {
        color: #a8b3c7;
    }

    /* ---------- INPUT ---------- */

    .stTextInput input {
        background-color: #111827 !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 10px !important;
    }

    .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #4f46e5;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        min-height: 42px;
    }

    .stButton > button:hover {
        background-color: #6366f1;
        border-color: #6366f1;
        color: white;
    }

    /* ---------- METRICS ---------- */

    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #27324a;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
    }

    /* ---------- CHAT ---------- */

    [data-testid="stChatMessage"] {
        border: 1px solid #27324a;
        border-radius: 12px;
        background-color: #111827;
        margin-bottom: 12px;
    }

    /* ---------- DIVIDER ---------- */

    hr {
        border-color: #20283a;
    }

    /* ---------- CAPTION ---------- */

    .stCaption {
        color: #8b98ad !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv(
    "HUGGINGFACEHUB_ACCESS_TOKEN"
)

if not HF_TOKEN:
    st.error(
        "HUGGINGFACEHUB_ACCESS_TOKEN was not found "
        "in your .env file."
    )
    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "video_id" not in st.session_state:
    st.session_state.video_id = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_video_id(url):

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"
    ]

    for pattern in patterns:

        match = re.search(pattern, url)

        if match:
            return match.group(1)

    if re.fullmatch(
        r"[A-Za-z0-9_-]{11}",
        url.strip()
    ):
        return url.strip()

    return None


# ============================================================
# FETCH TRANSCRIPT
# ============================================================

def fetch_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        snippet.text
        for snippet in transcript_list
    )

    return transcript


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(transcript):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store, len(chunks)


# ============================================================
# CREATE RAG CHAIN
# ============================================================

def create_rag_chain(vector_store):

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    llm = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        task="text-generation",
        temperature=0.2
    )

    model = ChatHuggingFace(
        llm=llm
    )

    prompt = PromptTemplate(
        template="""
You are an intelligent YouTube video assistant.

Answer the user's question ONLY using
the provided video transcript context.

If the answer cannot be found in the
transcript, say:

"I don't know based on the provided transcript."

Do not invent information.

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=[
            "context",
            "question"
        ]
    )

    def format_docs(docs):

        return "\n\n".join(
            doc.page_content
            for doc in docs
        )

    parallel_chain = RunnableParallel(
        {
            "context": (
                retriever
                | RunnableLambda(format_docs)
            ),

            "question": RunnablePassthrough()
        }
    )

    main_chain = (
        parallel_chain
        | prompt
        | model
        | StrOutputParser()
    )

    return main_chain


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🎬 VideoMind AI")

    st.caption(
        "Your AI-powered YouTube research assistant."
    )

    st.divider()

    st.subheader("How it works")

    st.markdown("""
    **1. Input**

    Paste a YouTube video URL.

    **2. Transcript**

    Extract the video's transcript.

    **3. Chunking**

    Split the transcript into smaller chunks.

    **4. Embeddings**

    Convert text into semantic vectors.

    **5. Retrieval**

    FAISS finds relevant context.

    **6. Generation**

    The LLM generates a grounded answer.
    """)

    st.divider()

    st.subheader("Technology")

    st.caption("🔗 LangChain")
    st.caption("🔎 FAISS")
    st.caption("🤗 Hugging Face")
    st.caption("🎬 YouTube Transcript API")
    st.caption("🐍 Python")

    st.divider()

    if st.session_state.vector_store:

        st.success(
            "Knowledge base ready"
        )

    if st.button(
        "Clear Conversation"
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.title("🎬 VideoMind AI")

st.caption(
    "Turn YouTube videos into interactive AI knowledge bases."
)

st.write(
    "Analyze a video, retrieve its most relevant content, "
    "and ask questions using natural language."
)

st.divider()


# ============================================================
# VIDEO INPUT
# ============================================================

st.subheader("Analyze a YouTube Video")

col1, col2 = st.columns(
    [5, 1],
    vertical_alignment="bottom"
)

with col1:

    video_url = st.text_input(
        "YouTube URL",
        placeholder=(
            "Paste a YouTube video URL here..."
        ),
        label_visibility="visible"
    )

with col2:

    analyze_button = st.button(
        "Analyze Video",
        use_container_width=True
    )


# ============================================================
# ANALYZE
# ============================================================

if analyze_button:

    if not video_url:

        st.warning(
            "Please enter a YouTube video URL."
        )

    else:

        video_id = extract_video_id(
            video_url
        )

        if not video_id:

            st.error(
                "Invalid YouTube URL or video ID."
            )

        else:

            with st.status(
                "Analyzing video...",
                expanded=True
            ) as status:

                try:

                    st.write(
                        "📥 Fetching transcript..."
                    )

                    transcript = fetch_transcript(
                        video_id
                    )

                    st.write(
                        "✂️ Splitting transcript..."
                    )

                    vector_store, chunk_count = (
                        create_vector_store(
                            transcript
                        )
                    )

                    st.session_state.transcript = (
                        transcript
                    )

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.video_id = (
                        video_id
                    )

                    st.session_state.chunks = (
                        chunk_count
                    )

                    st.session_state.messages = []

                    status.update(
                        label="Video analyzed successfully",
                        state="complete",
                        expanded=False
                    )

                except TranscriptsDisabled:

                    status.update(
                        label="Transcript unavailable",
                        state="error"
                    )

                    st.error(
                        "This video has transcripts disabled."
                    )

                except Exception as e:

                    status.update(
                        label="Analysis failed",
                        state="error"
                    )

                    st.error(str(e))


# ============================================================
# KNOWLEDGE BASE INFORMATION
# ============================================================

if st.session_state.vector_store:

    st.divider()

    st.subheader(
        "Video Knowledge Base"
    )

    transcript_length = len(
        st.session_state.transcript
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Transcript",
            "Loaded"
        )

    with col2:

        st.metric(
            "Text Chunks",
            st.session_state.chunks
        )

    with col3:

        st.metric(
            "Characters",
            f"{transcript_length:,}"
        )

    with col4:

        st.metric(
            "Vector Store",
            "FAISS"
        )

    st.success(
        "AI knowledge base is ready. "
        "You can now ask questions about the video."
    )


# ============================================================
# CHAT
# ============================================================

st.divider()

st.subheader(
    "💬 Ask About the Video"
)


if not st.session_state.vector_store:

    st.info(
        "Paste a YouTube URL above and click "
        "**Analyze Video** to get started."
    )

else:

    # Previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask anything about the video..."
    )

    if question:

        # User message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # Assistant response

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the video..."
            ):

                try:

                    chain = create_rag_chain(
                        st.session_state.vector_store
                    )

                    answer = chain.invoke(
                        question
                    )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"Something went wrong: {e}"
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "VideoMind AI • Built with Python, LangChain, "
    "FAISS, Hugging Face and Streamlit"
)