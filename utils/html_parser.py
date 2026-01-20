from bs4 import BeautifulSoup
from config.settings import HTML_REMOVE_TAGS


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    for element in soup(HTML_REMOVE_TAGS):
        element.decompose()
    return soup.get_text(separator='\n', strip=True)