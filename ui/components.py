import gradio as gr
from config.settings import DEFAULT_CRAWL_DEPTH, MAX_CRAWL_DEPTH


def create_loader_config():
    with gr.Row():
        loader_type = gr.Radio(
            choices=["Simple Web Loader", "Recursive Crawler", "Sitemap"],
            value="Recursive Crawler",
            label="Loading Method",
            info="Recursive Crawler follows links automatically. Sitemap loads all pages from sitemap.xml"
        )
        max_depth = gr.Slider(
            minimum=1,
            maximum=MAX_CRAWL_DEPTH,
            value=DEFAULT_CRAWL_DEPTH,
            step=1,
            label="Crawl Depth (for Recursive Crawler)",
            info="How many levels deep to follow links. 2-3 is usually enough."
        )
    return loader_type, max_depth


def create_url_input():
    return gr.Textbox(
        label="Documentation URLs",
        placeholder="https://docs.python.org/3/tutorial/\nhttps://fastapi.tiangolo.com/\nhttps://react.dev/learn",
        lines=5
    )


def create_chatbot():
    return gr.Chatbot(
        label="💬 Code Assistant",
        height=520
    )


def create_question_input():
    return gr.Textbox(
        label="Your Question",
        placeholder="How do I create a REST API endpoint?",
        lines=2
    )


def create_example_questions():
    return [
        ["How do I read a CSV file?"],
        ["Show me how to create a class in Python"],
        ["How do I handle async operations?"],
        ["What's the best way to validate user input?"],
        ["Show me the latest syntax for this feature"],
    ]


def get_loading_guide_markdown():
    return """
### 📖 Loading Method Guide:

**🔹 Simple Web Loader**
- Loads only the exact URL you provide
- Use when: You have a single page or know all URLs
- Speed: Fast

**🔹 Recursive Crawler** (Recommended)
- Follows all links on the page
- Use when: Documentation has multiple interconnected pages
- Depth 2: Main page + all linked pages + their links
- Depth 3: Goes one level deeper (slower but more complete)
- Speed: Medium

**🔹 Sitemap**
- Loads from sitemap.xml (fastest for large sites)
- Use when: Site has a sitemap (most modern docs do)
- Automatically finds all pages
- Speed: Fastest for large sites

### 🌐 Popular Documentation URLs:
- Python: `https://docs.python.org/3/tutorial/`
- FastAPI: `https://fastapi.tiangolo.com/tutorial/`
- React: `https://react.dev/learn`
- Django: `https://docs.djangoproject.com/en/stable/intro/`
- Flask: `https://flask.palletsprojects.com/en/latest/quickstart/`
- LangChain: `https://python.langchain.com/docs/get_started/introduction`
"""


def get_info_markdown():
    """Get system info and tips markdown"""
    return """
### 🎯 How This Works
1. **Crawling**: Automatically follows links in documentation to load complete content
2. **Chunking**: Splits documents into overlapping pieces (1500 chars with 300 char overlap)
3. **Embedding**: Converts text to vectors using all-mpnet-base-v2 model
4. **Retrieval**: Uses MMR to find 8 most relevant & diverse chunks
5. **Generation**: DeepSeek Coder generates answers based only on retrieved docs

### ✨ Accuracy Improvements in This Version
- ✅ **Smart Crawling**: Follows all documentation links automatically
- ✅ **Better Embeddings**: Upgraded to all-mpnet-base-v2 (more accurate)
- ✅ **Larger Chunks**: 1500 chars with 300 overlap (preserves more context)
- ✅ **MMR Retrieval**: Gets diverse results, not just similar ones
- ✅ **More Context**: Retrieves 8 chunks instead of 4
- ✅ **Strict Prompt**: Instructs model to use ONLY documentation
- ✅ **Source Tracking**: Shows which documents were used
- ✅ **Zero Temperature**: More deterministic, less creative (more accurate)

### 💡 Tips for Best Results
1. **Use Recursive Crawler** with depth 2-3 for interconnected docs
2. **Use Sitemap** for large documentation sites (fastest)
3. **Be Specific** in your questions - mention version or framework
4. **Check Sources** - the model cites which parts of docs it used
5. **Reload Docs** if you suspect outdated information

### 🔧 Troubleshooting
- **Slow loading?** Reduce crawl depth or use fewer URLs
- **Incomplete results?** Increase crawl depth to 3
- **Out of memory?** Close other GPU apps or reduce docs
- **Wrong info?** Check if docs were actually loaded (see chunk count)
- **Sitemap fails?** Switch to Recursive Crawler mode
"""