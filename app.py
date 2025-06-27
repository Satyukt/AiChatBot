
import streamlit as st
import os
import tempfile
import pdfplumber
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.documents import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not openai_api_key:
    st.error("OpenAI API key not found. Please set it in the .env file.")
    st.stop()

st.markdown(
    """
    <style>
    .welcome-box {
        background-color: #f0f9ff;
        border-radius: 15px;
        padding: 25px;
        margin-top: 20px;
        border: 1px solid #d1eaff;
        box-shadow: 0 0 8px rgba(0, 136, 255, 0.2);
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }
    .welcome-title {
        color: #007acc;
        font-size: 26px;
        font-weight: 600;
    }
    .welcome-sub {
        font-size: 17px;
        line-height: 1.6;
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #007acc;
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
        margin: 5px;
    }
    .stButton>button:hover {
        background-color: #005f99;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Welcome box content
st.markdown(
    '<div class="welcome-box"><div class="welcome-title">🌾 Welcome to Satyukt Analytics Virtual Assistant 🌾 </div></div>',
    unsafe_allow_html=True
)

# Interactive elements
with st.container():
    st.markdown('', unsafe_allow_html=True)

    st.markdown(
        '<div class="welcome-sub">Empowering Agriculture with Satellite Intelligence & AI 🚀</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="welcome-sub">🤖 I’m your smart assistant — ready to help with <strong>crop monitoring</strong>, '
        '<strong>insurance claims</strong>, <strong>risk analytics</strong>, and more.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="welcome-sub"><strong>Ask me about:</strong></div>', unsafe_allow_html=True)
    cols = st.columns(4)
    services = ["Sat2Farm", "Sat4Agri", "Sat4Risk", "Sat2Credit"]
    for i, service in enumerate(services):
        with cols[i]:
            if st.button(service):
                st.write(f"")

    with st.expander("🌍 Who We Serve"):
        st.markdown(
            '<div class="welcome-sub">Farmers, Agri-banks, Insurers & Governments</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="welcome-sub"><strong>Available Languages</strong></div>', unsafe_allow_html=True)
    languages = [
        "English", "हिंदी", "ಕನ್ನಡ", "தமிழ்", "తెలుగు", "বাংলা", "मराठी", "ગુજરાતી", "ਪੰਜਾਬੀ"
    ]
    selected_lang = st.selectbox("", languages, index=0)

    st.markdown('</div>', unsafe_allow_html=True)

prompt = ChatPromptTemplate.from_template(
    f"""
    You are a multilingual expert AI assistant. Use ONLY the information provided in the context (extracted from the PDF) to answer user questions.

Instructions:
1. Search the entire context thoroughly before responding.
2. Answer clearly and concisely in **{selected_lang}**.
3. If the answer is partially available, explain using what you found and clearly state the limitation.
4. If the answer is completely missing, reply in **{selected_lang}**: 
   "कृपया अधिक जानकारी के लिए 8043755513 पर संपर्क करें।" 
5. Never default to English unless the question is in English.
6. Never mention the context or PDF explicitly.

<context>
{{context}}
</context>
Question: {{input}}
    """
)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0.3)

def is_out_of_context(answer):
    keywords = [
        "i'm sorry", "i don't know", "not sure", "out of context",
        "invalid", "no mention"
    ]
    return any(k in answer.lower() for k in keywords)

def extract_text_with_pdfplumber(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def initialize_vector_db(pdf_file):
    if "vector_store" not in st.session_state:
        try:
            with st.spinner("Hang tight! We’re connecting you to our Virtual Assistant"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(pdf_file.read())
                    pdf_path = temp_file.name

                text_data = extract_text_with_pdfplumber(pdf_path)

                if not text_data.strip():
                    st.error("PDF appears empty or unreadable.")
                    return False

                doc = Document(page_content=text_data)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
                chunks = text_splitter.split_documents([doc])

                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/distiluse-base-multilingual-cased-v2',
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )

                st.session_state.vector_store = DocArrayInMemorySearch.from_documents(
                    chunks, st.session_state.embeddings
                )

                os.unlink(pdf_path)

            return True

        except Exception as e:
            st.error(f"Error: {str(e)}")
            return False
    return True

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Auto-load PDF from same folder instead of asking for upload
default_pdf_path = "SatyuktQueries.pdf"
if os.path.exists(default_pdf_path):
    class DummyFile:
        def read(self):
            with open(default_pdf_path, "rb") as f:
                return f.read()
    pdf_input_from_user = DummyFile()

    if initialize_vector_db(pdf_input_from_user):
        st.success("Hi there! 👋 How can I assist you today? Feel free to ask me anything About Us — I’m here to help You!")
else:
    st.error(f"PDF file '{default_pdf_path}' not found in the project directory.")

# Chat interface
if "vector_store" in st.session_state:
    st.subheader("Chat History")
    for msg in st.session_state.chat_history:
        role = "**You:**" if msg["role"] == "user" else "**Satyukt:**"
        st.write(f"{role} {msg['content']}")
        st.write("---")

    user_prompt = st.text_input("Enter Your Queries In Your Preferred language:")

    if st.button("Send"):
        if user_prompt:
            st.session_state.chat_history.append({"role": "user", "content": user_prompt})

            with st.spinner("Typing..."):
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 5})
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                response = retrieval_chain.invoke({'input': user_prompt})
                answer = response['answer']

                if is_out_of_context(answer):
                    answer = "Let me connect you with the right team for this! Drop a message to support@satyukt.com — they’ll take it from here. 🙌"

                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.experimental_rerun()
        else:
            st.error("Please enter a question.")
else:
    st.info("Waiting for Our Virtual Assistant...")
