import gradio as gr
import torch
from typing import List, Tuple
from core.document_loader import DocumentLoader
from core.embeddings import get_embeddings
from core.vectorstore import VectorStoreManager
from core.chain import RAGChain
from utils.html_parser import extract_text_from_html
from utils.chat_history import ChatHistoryStore
from ui.components import (
    create_loader_config,
    create_url_input,
    create_chatbot,
    create_question_input,
    create_example_questions,
    get_loading_guide_markdown,
    get_info_markdown
)
from config.settings import (
    DEVICE,
    CHAT_HISTORY_TTL,
    CHAT_HISTORY_LIMIT,
    CHAT_HISTORY_UI_LIMIT,
    SERVER_NAME,
    SERVER_PORT,
    SERVER_SHARE
)


class CodeAssistantUI:
    
    def __init__(self):
        self.assistant_ready = False
        self.doc_loader = DocumentLoader(extract_text_from_html)
        self.vectorstore_manager = VectorStoreManager()
        self.chain = None
        self.chat_store = ChatHistoryStore(ttl_seconds=CHAT_HISTORY_TTL)
    
    def load_docs(
        self, 
        urls_text: str, 
        max_depth: int, 
        loader_type: str, 
        progress=gr.Progress(), 
        reload: bool = False
    ) -> str:
        if reload:
            self.vectorstore_manager.delete_collection()
        
        try:
            progress(0, desc="Starting...")
            
            # Parse URLs
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            if not urls:
                return "❌ Please provide at least one URL"
            
            # Load embeddings
            progress(0.2, desc="Loading embedding model...")
            embeddings = get_embeddings()
            
            # Load documents
            progress(0.4, desc=f"Loading documents using {loader_type}...")
            try:
                documents = self.doc_loader.load_documents(urls, loader_type, max_depth)
            except Exception as e:
                return f"❌ Failed to load documents: {str(e)}"
            
            if not documents:
                return "❌ No documents loaded. Check your URLs."
            
            progress(0.6, desc=f"Loaded {len(documents)} documents")
            
            # Split into chunks
            progress(0.7, desc="Processing documents...")
            chunks = self.vectorstore_manager.split_documents(documents)
            
            # Create vector store
            progress(0.8, desc="Creating vector database...")
            self.vectorstore_manager.create_vectorstore(chunks, embeddings)
            
            # Setup chain
            progress(0.9, desc="Setting up AI model...")
            retriever = self.vectorstore_manager.get_retriever()
            self.chain = RAGChain(retriever, self.format_chat_history)
            
            self.assistant_ready = True
            progress(1.0, desc="Complete!")
            
            return (
                f"✅ Successfully loaded {len(documents)} documents, "
                f"created {len(chunks)} chunks.\n\n"
                f"Loader: {loader_type}\n"
                f"Depth: {max_depth if loader_type == 'Recursive Crawler' else 'N/A'}\n\n"
                f"You can now ask questions!"
            )
            
        except Exception as e:
            import traceback
            return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
    
    def format_chat_history(self) -> str:
        history = self.chat_store.load_recent(limit=CHAT_HISTORY_LIMIT)
        if not history:
            return "No previous conversation."
        
        formatted = []
        for q, a in history:
            formatted.append(f"User: {q}")
            formatted.append(f"Assistant: {a[:200]}...")
        return "\n".join(formatted)
    
    def load_chatbot_history(self) -> List[dict]:
        history = self.chat_store.load_recent(limit=CHAT_HISTORY_UI_LIMIT)
        messages = []
        for q, a in history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
        return messages
    
    def ask_question(
        self, 
        question: str, 
        chat_history: List[dict]
    ) -> Tuple[List[dict], str]:
        if not self.assistant_ready:
            chat_history.append({
                "role": "assistant", 
                "content": "⚠️ Please load documentation first."
            })
            return chat_history, ""
        
        if not question.strip():
            return chat_history, ""
        
        try:
            # Add user message
            chat_history.append({"role": "user", "content": question})
            
            # Get answer
            answer = self.chain.invoke(question)
            self.chat_store.save(question, answer)
            
            # Add assistant message
            chat_history.append({"role": "assistant", "content": answer})
            
            return chat_history, ""
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
            chat_history.append({"role": "assistant", "content": error_msg})
            return chat_history, ""
    
    def clear_history(self) -> List:
        if hasattr(self.chat_store, 'cleanup'):
            self.chat_store.cleanup()
        return []


def create_app() -> gr.Blocks:
    assistant = CodeAssistantUI()
    
    with gr.Blocks(title="RAG Code Assistant") as app:
        gr.Markdown("""
        # 🤖 RAG Code Assistant with Smart Crawling
        ### AI-powered coding help that crawls entire documentation sites
        **Model:** DeepSeek Coder 6.7B | **Embeddings:** all-mpnet-base-v2
        """)
        
        # Load Documentation Tab
        with gr.Tab("📚 Load Documentation"):
            gr.Markdown("### Step 1: Configure Document Loading")
            loader_type, max_depth = create_loader_config()
            
            gr.Markdown("### Step 2: Enter Documentation URLs")
            urls_input = create_url_input()
            
            with gr.Row():
                load_btn = gr.Button("🚀 Load & Crawl Documentation", variant="primary", size="lg")
                reload_btn = gr.Button("♻️ Reload Documentation", variant="secondary")
            
            load_output = gr.Textbox(label="Status", lines=5)
            
            load_btn.click(
                fn=assistant.load_docs,
                inputs=[urls_input, max_depth, loader_type],
                outputs=[load_output]
            )
            
            reload_btn.click(
                fn=lambda u, d, l: assistant.load_docs(u, d, l, reload=True),
                inputs=[urls_input, max_depth, loader_type],
                outputs=[load_output]
            )
            
            gr.Markdown(get_loading_guide_markdown())
        
        # Ask Questions Tab
        with gr.Tab("💬 Ask Questions"):
            gr.Markdown("### Step 2: Ask Coding Questions")
            chatbot = create_chatbot()
            question_input = create_question_input()
            
            with gr.Row():
                ask_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear History", variant="secondary")
            
            gr.Markdown("**Example Questions:**")
            gr.Examples(
                examples=create_example_questions(),
                inputs=question_input
            )
            
            # Event handlers
            ask_btn.click(
                fn=assistant.ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            question_input.submit(
                fn=assistant.ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            clear_btn.click(
                fn=assistant.clear_history,
                outputs=[chatbot]
            )
            
            # Load history on startup
            app.load(
                fn=assistant.load_chatbot_history,
                outputs=[chatbot]
            )
        
        # Settings & Info Tab
        with gr.Tab("⚙️ Settings & Info"):
            gr.Markdown("### System Information")
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
            device_text = f"Using: {DEVICE.upper()}"
            if device_name:
                device_text += f" ({device_name})"
            
            gr.Textbox(label="Device", value=device_text, interactive=False)
            gr.Markdown(get_info_markdown())
    
    return app


def launch_app():
    app = create_app()
    app.launch(
        server_name=SERVER_NAME,
        server_port=SERVER_PORT,
        share=SERVER_SHARE,
        theme=gr.themes.Soft()
    )