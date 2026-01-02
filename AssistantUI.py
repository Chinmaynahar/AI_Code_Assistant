

import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch


class CodeAssistantUI:
    def __init__(self):
        self.assistant_ready = False
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_docs(self, urls_text, progress=gr.Progress()):
        """Load documentation from URLs"""
        try:
            progress(0, desc="Starting...")
            
            # Parse URLs
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            if not urls:
                return "❌ Please provide at least one URL"
            
            # Load embeddings
            progress(0.2, desc="Loading embedding model...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': self.device}
            )
            
            # Load documents
            progress(0.4, desc=f"Loading {len(urls)} URLs...")
            documents = []
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    documents.extend(loader.load())
                except Exception as e:
                    return f"❌ Failed to load {url}: {str(e)}"
            
            # Split into chunks
            progress(0.6, desc="Processing documents...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            progress(0.8, desc="Creating vector database...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./gradio_db"
            )
            
            # Setup chain using LCEL
            progress(0.9, desc="Setting up AI model...")
            llm = Ollama(model="deepseek-coder:6.7b", temperature=0.1)
            
            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # Create prompt template
            prompt_template = """You are an expert code assistant. Use the following documentation to answer the question.
Provide clear code examples with comments when relevant.

Documentation:
{context}

Question: {question}

Answer:"""

            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Format retrieved documents
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            # Create the chain using LCEL
            self.chain = (
                {
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough()
                }
                | prompt
                | llm
                | StrOutputParser()
            )
            
            self.assistant_ready = True
            progress(1.0, desc="Complete!")
            
            return f"✅ Successfully loaded {len(documents)} documents, created {len(chunks)} chunks.\n\nYou can now ask questions!"
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def ask_question(self, question):
        """Answer a coding question"""
        if not self.assistant_ready:
            return "⚠️ Please load documentation first!", ""
        
        if not question.strip():
            return "⚠️ Please enter a question", ""
        
        try:
            # Get relevant documents for sources
            source_docs = self.retriever.invoke(question)
            
            # Get answer from chain
            answer = self.chain.invoke(question)
            
            # Format sources
            sources = "\n\n📚 **Sources:**\n"
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                preview = doc.page_content[:100].replace('\n', ' ')
                sources += f"{i}. {source}\n   Preview: {preview}...\n"
            
            return answer, sources
            
        except Exception as e:
            return f"❌ Error: {str(e)}", ""


# Initialize assistant
assistant = CodeAssistantUI()

# Create Gradio interface
with gr.Blocks(title="RAG Code Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 RAG Code Assistant
    ### Free AI-powered coding help using your documentation
    
    **GPU:** RTX 4050 | **Model:** DeepSeek Coder 6.7B
    """)
    
    with gr.Tab("📚 Load Documentation"):
        gr.Markdown("""
        ### Step 1: Add Documentation URLs
        Enter one URL per line. The assistant will learn from these docs.
        """)
        
        urls_input = gr.Textbox(
            label="Documentation URLs",
            placeholder="https://docs.python.org/3/tutorial/\nhttps://fastapi.tiangolo.com/\nhttps://react.dev/learn",
            lines=5
        )
        
        load_btn = gr.Button("🚀 Load Documentation", variant="primary", size="lg")
        load_output = gr.Textbox(label="Status", lines=3)
        
        load_btn.click(
            fn=assistant.load_docs,
            inputs=[urls_input],
            outputs=[load_output]
        )
        
        gr.Markdown("""
        **Popular Documentation URLs:**
        - Python: `https://docs.python.org/3/tutorial/`
        - FastAPI: `https://fastapi.tiangolo.com/tutorial/`
        - React: `https://react.dev/learn`
        - Django: `https://docs.djangoproject.com/en/stable/intro/`
        - Flask: `https://flask.palletsprojects.com/en/latest/quickstart/`
        """)
    
    with gr.Tab("💬 Ask Questions"):
        gr.Markdown("### Step 2: Ask Coding Questions")
        
        with gr.Row():
            with gr.Column():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="How do I create a REST API endpoint?",
                    lines=3
                )
                ask_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                
                # Example questions
                gr.Markdown("**Example Questions:**")
                examples = gr.Examples(
                    examples=[
                        ["How do I read a CSV file?"],
                        ["Show me how to create a class in Python"],
                        ["How do I handle async operations?"],
                        ["What's the best way to validate user input?"],
                    ],
                    inputs=question_input
                )
        
        answer_output = gr.Textbox(label="💡 Answer", lines=15)
        sources_output = gr.Textbox(label="📚 Sources", lines=5)
        
        ask_btn.click(
            fn=assistant.ask_question,
            inputs=[question_input],
            outputs=[answer_output, sources_output]
        )
    
    with gr.Tab("⚙️ Settings"):
        gr.Markdown("""
        ### System Information
        """)
        
        device_info = gr.Textbox(
            label="Device",
            value=f"Using: {assistant.device.upper()}" + 
                  (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""),
            interactive=False
        )
        
        gr.Markdown("""
        ### Tips for Best Results
        
        1. **Load Quality Documentation**: Use official docs from framework websites
        2. **Be Specific**: Ask detailed questions about what you want to accomplish
        3. **Iterate**: If answer isn't helpful, rephrase your question
        4. **Check Sources**: Review the source documents for more context
        
        ### Troubleshooting
        
        - **Slow responses?** Close other GPU-intensive apps
        - **Out of memory?** Use fewer documentation URLs
        - **Ollama not responding?** Run `ollama serve` in terminal
        """)

if __name__ == "__main__":
    print("🚀 Starting RAG Code Assistant Web UI...")
    print(f"📊 Device: {assistant.device.upper()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print("\n🌐 Opening web interface...")
    
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False  # Set to True to create public link
    )