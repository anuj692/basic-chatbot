import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# ---- API Keys ----
HF_TOKEN = "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"
GROQ_API_KEY = "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"  # Get free at console.groq.com

# ---- Groq query function ----
def query_groq(context, question):
    client = Groq(api_key="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")
    result = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # ✅ active model
        messages=[
            {
                "role": "system",
               "content": (
    "You are an expert analytical assistant. Read the provided context carefully. "
    "Answer the user's question by synthesizing, summarizing, and making logical "
    "inferences STRICTLY based on the information provided in the context. "
    "You can connect different parts of the text to form a complete answer. "
    "Do not just copy-paste lines; explain the concepts naturally and in detail. "
    "If the underlying information needed to answer the question is completely "
    "missing from the context, politely say 'I cannot find the answer in the provided text.'"
)
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=512,
        temperature=0.1
    )
    return result.choices[0].message.content

# ---- Streamlit App ----
st.set_page_config(page_title="Custom Text RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Custom Text RAG Chatbot")
st.markdown("Paste your text, process it, and ask questions based on the content.")

# Sidebar
st.sidebar.header("📝 Text Input")
text = st.sidebar.text_area(
    "Paste your paragraph, article, or any large block of text here:",
    height=400,
    placeholder="Enter your text here..."
)

if st.sidebar.button("🚀 Process Text", type="primary"):
    if text.strip():
        with st.spinner("Processing text (Structure-Based Chunking)..."):
            
            # Here we define the priority of separators
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150,
                separators=["\n\n", "\n", ". ", " ", ""] # Respects paragraphs, then lines, then sentences
            )
            chunks = text_splitter.split_text(text)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_texts(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.chunks_count = len(chunks)

        st.sidebar.success(f"✅ Text processed! Created {len(chunks)} structured chunks.")
    else:
        st.sidebar.error("❌ Please enter some text to process.")

# Chat interface
st.header("💬 Chat Interface")

if "vectorstore" in st.session_state:
    st.info(f"📊 Vector database ready with {st.session_state.chunks_count} text chunks.")

    user_question = st.text_input(
        "Ask a question about the processed text:",
        placeholder="What would you like to know about the text?"
    )

    if st.button("🔍 Ask", type="primary") and user_question.strip():
        with st.spinner("Thinking..."):
            try:
                # Retrieve relevant chunks (k=4 for better context window)
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )
                source_docs = retriever.invoke(user_question)

                # Build context from retrieved chunks
                context = "\n\n".join(doc.page_content for doc in source_docs)

                # Query Groq
                answer = query_groq(context, user_question)

                # Display answer
                st.subheader("🤖 Answer:")
                st.write(answer)

                # Display source chunks
                with st.expander("📄 Source Text Chunks Used:"):
                    for i, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(doc.page_content)
                        st.markdown("---")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("💡 Make sure your GROQ_API_KEY is correct.")
else:
    st.warning("⚠️ Please process some text first using the sidebar.")

st.markdown("---")
st.markdown("*Built with Streamlit, LangChain, FAISS, and Groq (Llama 3.1).*")