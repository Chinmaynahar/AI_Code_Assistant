from typing import List, Callable
from langchain_community.document_loaders import (
    WebBaseLoader,
    RecursiveUrlLoader,
    SitemapLoader
)
from langchain_core.documents import Document
from config.settings import SITEMAP_PATHS, CRAWLER_TIMEOUT


class DocumentLoader:
    def __init__(self, text_extractor: Callable[[str], str]):
        self.text_extractor = text_extractor
    
    def load_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        return loader.load()
    
    def load_recursive(self, url: str, max_depth: int) -> List[Document]:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=max_depth,
            extractor=self.text_extractor,
            prevent_outside=True,
            use_async=False,
            timeout=CRAWLER_TIMEOUT
        )
        return loader.load()
    
    def load_from_sitemap(self, url: str) -> List[Document]:
        if url.endswith('sitemap.xml'):
            sitemap_urls = [url]
        else:
            base_url = url.rstrip('/')
            sitemap_urls = [f"{base_url}{path}" for path in SITEMAP_PATHS]
        
        for sitemap_url in sitemap_urls:
            try:
                loader = SitemapLoader(web_path=sitemap_url)
                docs = loader.load()
                if docs:
                    return docs
            except Exception:
                continue
        
        raise Exception(f"Could not find valid sitemap at {url}")
    
    def load_documents(
        self, 
        urls: List[str], 
        loader_type: str, 
        max_depth: int = 2
    ) -> List[Document]:
        all_documents = []
        
        for url in urls:
            if loader_type == "Sitemap":
                docs = self.load_from_sitemap(url)
            elif loader_type == "Recursive Crawler":
                docs = self.load_recursive(url, max_depth)
            else:  # Simple Web Loader
                docs = self.load_from_url(url)
            
            all_documents.extend(docs)
        
        return all_documents