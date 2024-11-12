import os
import time
import requests
from bs4 import BeautifulSoup
import streamlit as st
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from validators import url as validate_url

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from langchain.embeddings import HuggingFaceEmbeddings
from langchain_cerebras import ChatCerebras

from langchain.vectorstores import Weaviate
import weaviate

# Initialize abbreviation dictionary
abbreviation_dict = {
    'SCHS': 'Silver Creek High School',
    'PHHS': 'Piedmont Hills High School',
    'EVHS': 'Evergreen Valley High School',
    'IHS': 'Independence High School',
    'ESUHSD': 'East Side Union High School District',
    # Add more abbreviations as needed
}

def replace_abbreviations(text):
    for abbr, full in abbreviation_dict.items():
        text = text.replace(abbr, full)
    return text

def extract_text(soup):
    # Only extract text from specific tags
    content = ''
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'li']):
        content += tag.get_text(separator=' ', strip=True) + ' '
    return content

def crawl_website(base_url, max_pages, allowed_paths):
    visited = set()
    to_visit = [base_url]
    all_texts = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url)
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract visible text from the page
            text = extract_text(soup)
            all_texts.append(text)
            visited.add(current_url)

            # Find and enqueue links from the current page
            for link in soup.find_all("a", href=True):
                full_url = urljoin(base_url, link["href"])
                parsed_base = urlparse(base_url)
                parsed_full = urlparse(full_url)

                # Check if the URL is within the same domain and allowed paths
                if parsed_full.netloc == parsed_base.netloc and any(path in full_url for path in allowed_paths):
                    if full_url not in visited and full_url not in to_visit:
                        to_visit.append(full_url)

            time.sleep(1)  # Crawl delay to respect server load

        except Exception as e:
            st.error(f"Error scraping {current_url}: {e}")

    return all_texts

def reset_session_state(website):
    st.session_state.messages = []
    st.session_state.processed_links = set()
    st.session_state.docsearch = None
    st.session_state.embeddings = None
    st.session_state.last_website = website

def upload_vectors(docs, embeddings, client, class_name):
    # Initialize Weaviate vector store
    vector_store = Weaviate(client=client, index_name=class_name, text_key="text", embedding=embeddings)

    # Add documents to vector store
    with st.spinner("Indexing website content..."):
        vector_store.add_documents(docs)

    return vector_store

# Streamlit UI setup
st.set_page_config(page_icon="🤖", layout="wide", page_title="Cerebras x Weaviate Demo")
st.subheader("Navigate Your School!", anchor=False)

# Sidebar settings and inputs
with st.sidebar:
    st.title("Settings")
    st.markdown("### Enter your Cerebras API Key below")
    CEREBRAS_API_KEY = st.text_input("Cerebras API Key:", type="password",value="csk-xnrcwxxnk9wt2vwj3d448m8jwfc2wpyd8c98mfknx38ryw3c")
    st.markdown("### Enter your Weaviate URL below")
    WEAVIATE_URL = st.text_input("Weaviate URL:", type="password",value="https://skcebcqytu25npjx6ghnma.c0.us-west3.gcp.weaviate.cloud")
    st.markdown("### Enter your Weaviate API Key below")
    WEAVIATE_API_KEY = st.text_input("Weaviate API Key:", type="password",value="PLU2ZntUMfqNBI2zw2cQmUafjJvaOODPOS2d")
    st.markdown("[Get your Cerebras API Key Here](https://inference.cerebras.ai/)")

    # Max pages slider for crawling
    max_pages = st.slider("Max Pages to Crawl", min_value=1, max_value=100, value=10)

if not CEREBRAS_API_KEY or not WEAVIATE_URL or not WEAVIATE_API_KEY:
    st.markdown("""
    ## Welcome to the Cerebras x Weaviate Demo!

    This website analysis tool allows you to input a school website and then ask questions about its content through vector storage with Weaviate and a custom LLM implementation with Cerebras.

    To get started:
    1. **Enter your Cerebras and Weaviate API credentials in the sidebar.**
    2. **Enter a website to analyze.**
    3. **Ask about it!**

    """)
    st.stop()

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_links" not in st.session_state:
    st.session_state.processed_links = set()
if "docsearch" not in st.session_state:
    st.session_state.docsearch = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "last_website" not in st.session_state:
    st.session_state.last_website = ''

# Get the website input
website = st.text_input("School Website:", placeholder="e.g., https://www.silvercreekhigh.org")

st.divider()

# Display chat messages stored in history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '❔'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if not website:
    st.markdown("Please enter a website.")
elif not validate_url(website):
    st.error("Please enter a valid URL.")
else:
    # Reset session state if a new website is entered
    if website != st.session_state.last_website:
        reset_session_state(website)

    if website in st.session_state.processed_links:
        st.markdown(f"The website **{website}** has already been processed. You can ask questions about it.")
    else:
        # Crawl the website and collect content
        allowed_paths = ['/about/', '/events/', '/news/', '/calendar/', '/home/', '/academics/', '/students/', '/parents/']

        with st.spinner(f"Crawling {website}..."):
            texts = crawl_website(website, max_pages, allowed_paths)

        # Replace abbreviations in texts
        texts = [replace_abbreviations(text) for text in texts]

        # Split the texts into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents([Document(page_content=text) for text in texts])

        # Load embeddings
        with st.spinner("Loading embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.embeddings = embeddings  # Store embeddings in session state

        # Connect to Weaviate
        client = weaviate.Client(
            url=WEAVIATE_URL,
            auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
            additional_headers={"X-OpenAI-Api-Key": WEAVIATE_API_KEY},
        )

        # Define Weaviate schema
        class_name = "SchoolContent"
        schema = {
            "classes": [
                {
                    "class": class_name,
                    "vectorizer": "none",  # We'll provide our own embeddings
                    "properties": [
                        {
                            "name": "text",
                            "dataType": ["text"],
                        },
                    ],
                }
            ]
        }

        # Create schema if it doesn't exist
        if not client.schema.contains(schema):
            client.schema.create(schema)

        # Upload vectors to Weaviate
        st.session_state.docsearch = upload_vectors(docs, embeddings, client, class_name)
        st.session_state.processed_links.add(website)
        st.session_state.messages = []

    # Ensure embeddings are available
    if "embeddings" not in st.session_state or st.session_state.embeddings is None:
        with st.spinner("Loading embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            st.session_state.embeddings = embeddings
    else:
        embeddings = st.session_state.embeddings

    if prompt := st.chat_input("Enter your question here..."):
        # Replace abbreviations in the prompt
        prompt = replace_abbreviations(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar='❔'):
            st.markdown(prompt)

        # Initialize the language model
        llm = ChatCerebras(api_key=CEREBRAS_API_KEY, model="llama3.1-8b")

        # Create a custom prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert assistant specializing in providing detailed information about schools.
            Based on the context provided, thoroughly answer the following question.

            Context:
            {context}

            Question:
            {question}

            Detailed Answer:
            """
        )

        # Load the QA chain with the custom prompt
        chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt_template)

        # Compute the embedding of the query
        query_embedding = embeddings.embed_query(prompt)

        # Perform similarity search using nearVector
        docs = st.session_state.docsearch.similarity_search_by_vector(query_embedding, k=4)

        # Generate the response
        response = chain.run(input_documents=docs, question=prompt)

        # Display the response
        with st.chat_message("assistant", avatar="🤖"):
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
