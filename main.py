import os
import time
import weaviate
import streamlit as st
from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cerebras import ChatCerebras
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from langchain.schema import Document
from urllib.parse import urljoin, urlparse

# Configure Chrome to run in headless mode (no pop-up window)
def get_headless_driver():
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")  # Disable GPU rendering
    options.add_argument("--no-sandbox")  # Bypass OS security model (for some environments)
    options.add_argument("--disable-dev-shm-usage")  # Avoid shared memory issues
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Function to scrape all content from a single page
def scrape_page(url):
    driver = get_headless_driver()
    driver.get(url)
    time.sleep(3)  # Adjust sleep time for page load if needed
    html_content = driver.page_source
    driver.quit()
    soup = BeautifulSoup(html_content, "html.parser")
    texts = [element.get_text(strip=True) for element in soup.find_all(True)]
    return texts

# Crawling function to find and process subpages (with max page limit)
def crawl_website(base_url, max_pages):
    visited = set()
    to_visit = [base_url]
    all_texts = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            texts = scrape_page(current_url)
            all_texts.extend(texts)
            visited.add(current_url)

            # Find and enqueue links from the current page
            soup = BeautifulSoup('\n'.join(texts), "html.parser")
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                if urlparse(full_url).netloc == urlparse(base_url).netloc:  # Stay within the same domain
                    to_visit.append(full_url)

        except Exception as e:
            st.error(f"Error scraping {current_url}: {e}")

    return all_texts

# Upload vectors to Weaviate
def upload_vectors(texts, embeddings, progress_bar, client):
    vector_store = WeaviateVectorStore(client=client, index_name="my_class", text_key="text", embedding=embeddings)
    
    # No need to access 'page_content' here since 'texts' are already strings.
    for i, text in enumerate(texts):
        vector_store.add_texts([text])  # Add text directly

        # Update progress bar
        progress_bar.progress((i + 1) / len(texts), "Indexing Website content... (this may take a bit)")

    progress_bar.empty()
    return vector_store

# Streamlit UI setup
st.set_page_config(page_icon="🤖", layout="wide", page_title="Cerebras")
st.subheader("Navigate your School!", divider="orange", anchor=False)

# Sidebar settings and inputs
with st.sidebar:
    st.title("Settings")
    st.markdown("### :red[Enter your Cerebras API Key below]")
    CEREBRAS_API_KEY = st.text_input("Cerebras API Key:", type="password")
    st.markdown("### :red[Enter your Weaviate URL below]")
    WEAVIATE_URL = st.text_input("Weaviate URL:", type="password")
    st.markdown("### :red[Enter your Weaviate API Key below]")
    WEAVIATE_API_KEY = st.text_input("Weaviate API Key:", type="password")
    st.markdown("[Get your Cerebras API Key Here](https://inference.cerebras.ai/)")

    # Max pages slider for crawling
    max_pages = st.slider("Max Pages to Crawl", min_value=1, max_value=100, value=10)

if not CEREBRAS_API_KEY or not WEAVIATE_URL or not WEAVIATE_API_KEY:
    st.markdown("""
    ## Welcome to Cerebras x Weaviate Demo!

    This Website analysis tool receives a site and allows you to ask questions about the content of it through vector storage with Weaviate and a custom LLM implementation with Cerebras.

    To get started:
    1. :red[Enter your Cerebras and Weaviate API credentials in the sidebar.]
    2. Enter a Website to Analyze.
    3. Ask about it!

    """)
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "website" not in st.session_state:
    st.session_state.website = ""
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None
if "processed_links" not in st.session_state:
    st.session_state.processed_links = set()

# Get the website input
website = st.text_input("School Website: ")

st.divider()

# Display chat messages stored in history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '❔'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if not website:
    st.markdown("Please enter a website.")
else:
    if website in st.session_state.processed_links:
        st.markdown(f"The website **{website}** has already been processed. You can ask questions about it.")
    else:
        texts = crawl_website(website, max_pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents([Document(page_content=t) for t in texts])

        with st.spinner("Loading embeddings..."):
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=WEAVIATE_URL,
            auth_credentials=weaviate.AuthApiKey(WEAVIATE_API_KEY),
        )

        progress_bar = st.progress(0, "Indexing Website content...")
        st.session_state.docsearch = upload_vectors([doc.page_content for doc in docs], embeddings, progress_bar, client)
        st.session_state.processed_links.add(website)
        st.session_state.messages = []

    if prompt := st.chat_input("Enter your prompt here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='❔'):
            st.markdown(prompt)

        llm = ChatCerebras(api_key=CEREBRAS_API_KEY, model="llama3.1-8b")
        chain = load_qa_chain(llm, chain_type="stuff")
        docs = st.session_state.docsearch.similarity_search(prompt)

        response = chain.run(input_documents=docs, question=prompt)
        with st.chat_message("assistant", avatar="🤖"):
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)