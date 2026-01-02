
import os
from langchain_community.document_loaders import WebBaseLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import torch

class CodeAssistant:
    def __init__(self, persist_directory="./code_assistant_db"):
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.chain = None
        

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
    
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': self.device}
        )
        
  
        print("Connecting to Ollama...")
        self.llm = Ollama(
            model="deepseek-coder:6.7b", 
            temperature=0.1
        )
        
    def load_documentation(self, urls=None, local_paths=None):
        """Load documentation from URLs or local files"""
        documents = []
        
     
        if urls:
            print(f"Loading {len(urls)} documentation URLs...")
            for url in urls:
                try:
                    loader = WebBaseLoader(url)
                    documents.extend(loader.load())
                    print(f"✓ Loaded: {url}")
                except Exception as e:
                    print(f"✗ Failed to load {url}: {e}")
        
     
        if local_paths:
            print(f"Loading local documentation...")
            for path in local_paths:
                try:
                    loader = DirectoryLoader(path, glob="**/*.md")
                    documents.extend(loader.load())
                    print(f"✓ Loaded: {path}")
                except Exception as e:
                    print(f"✗ Failed to load {path}: {e}")
        
        if not documents:
            raise ValueError("No documents loaded!")
        
 
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
     
        print("Creating vector database...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("✓ Vector database created!")
        
     
        self._setup_chain()
        
    def load_existing_db(self):
        """Load previously created vector database"""
        if not os.path.exists(self.persist_directory):
            raise ValueError(f"Database not found at {self.persist_directory}")
        
        print("Loading existing vector database...")
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        self._setup_chain()
        print("✓ Database loaded!")
        
    def _setup_chain(self):
        """Setup the RAG chain using LCEL (LangChain Expression Language)"""
        
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4} 
        )
        
    
        prompt_template = """You are an expert code assistant. Use the following documentation to answer the coding question.
If you don't know the answer, say so - don't make up information.
Provide code examples when relevant, with proper formatting and comments.

Documentation Context:
{context}

Question: {question}

Answer with clear explanation and code examples:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
      
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
   
        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
   
        self.retriever = retriever
        
    def ask(self, question):
        """Ask a coding question"""
        if not self.chain:
            raise ValueError("Please load documentation first!")
        
        print(f"\n Question: {question}\n")
        print(" Searching documentation...")
        
      
        source_docs = self.retriever.invoke(question)
        
      
        answer = self.chain.invoke(question)
        
        print("\n Answer:")
        print(answer)
        
        print("\n Sources:")
        for i, doc in enumerate(source_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            print(f"{i}. {source}")
        
        return {
            "answer": answer,
            "source_documents": source_docs
        }


# Example usage
if __name__ == "__main__":
    # Initialize assistant
    assistant = CodeAssistant()
    
    # Option 1: Load new documentation
    print("=" * 60)
    print("RAG CODE ASSISTANT - SETUP")
    print("=" * 60)
    
    # Example documentation URLs (replace with your own)
    docs_urls = [
        "https://docs.python.org/3/tutorial/index.html",
        "https://fastapi.tiangolo.com/tutorial/",
        # Add more URLs as needed
    ]
    
    # Load documentation (comment out if loading existing DB)
    assistant.load_documentation(urls=docs_urls)
    
    # Option 2: Load existing database (uncomment to use)
    # assistant.load_existing_db()
    
    # Ask questions
    print("\n" + "=" * 60)
    print("READY! You can now ask coding questions.")
    print("=" * 60)
    
    # Example questions
    questions = [
        "How do I create a REST API endpoint in FastAPI?",
        "Show me how to read a CSV file in Python",
        "How do I handle exceptions in Python?"
    ]
    
    for question in questions:
        assistant.ask(question)
        print("\n" + "-" * 60 + "\n")