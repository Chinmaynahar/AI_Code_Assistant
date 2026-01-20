from typing import List, Callable
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from config.settings import LLM_MODEL, LLM_TEMPERATURE


PROMPT_TEMPLATE = """You are an expert code assistant. Use ONLY the documentation provided below to answer questions.

IMPORTANT INSTRUCTIONS:
- Base your answer ONLY on the documentation context provided
- Do NOT use knowledge from your training if it contradicts the documentation
- If the documentation doesn't contain the answer, say "I couldn't find this information in the provided documentation"
- When providing code examples, use the exact syntax and patterns from the documentation
- Cite which part of the documentation you're using

Conversation history:
{chat_history}

Documentation Context:
{context}

User Question: {question}

Your Answer (based only on the documentation above):"""


def format_docs(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content
        formatted.append(f"[Source {i}: {source}]\n{content}")
    return "\n\n---\n\n".join(formatted)


class RAGChain:
    
    def __init__(self, retriever, chat_history_formatter: Callable[[], str]):
        self.llm = Ollama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        self.retriever = retriever
        self.chat_history_formatter = chat_history_formatter
        
        self.prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "question", "chat_history"]
        )
        
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self.chat_history_formatter()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def invoke(self, question: str) -> str:
        return self.chain.invoke(question)