# 🎓 ClassmateAI  

Welcome to **ClassmateAI** — your friendly digital assistant for navigating school life! Helps if you're a student trying to settle into a new school or need information about the school in general.

---

## 💡 What is ClassmateAI?

**ClassmateAI** is an intelligent assistant designed to help students with everything related to schoo, from understanding your new school’s website to answering questions about its content.

ClassmateAI can:
- 🗂 **Explore and understand your school’s resources** easily.
- 💬 **Get quick answers** by interacting with ClassmateAI’s chat interface.
- 📑 **Analyze school websites** and store key information in a searchable knowledge base.
- 🎯 **Save time** by indexing and searching through documents effortlessly.  
- 🔍 **Ask questions**, and get precise answers powered by smart embeddings and LLMs.

---

## 🚀 Features

- **Smart Crawling** 🕵️‍♂️: Analyze any school website, breaking down pages to find the information you need.
- **Chat-Based Interaction** 🤖: Ask questions directly through our chat interface, and get answers powered by Cerebras models.
- **Efficient Vector Search** 📊: Stores and retrieves website content using Weaviate for lightning-fast responses.
- **Headless Scraping** 🦾: No browser pop-ups! ClassmateAI works seamlessly in the background using headless Chrome.
- **Minimal Code** 🧑‍💻: Less than **250 lines of code** — small but mighty! 

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit  
- **Backend:** Python, Selenium, BeautifulSoup  
- **Vector Search:** Weaviate  
- **Embeddings:** SentenceTransformer & Cerebras Models  
- **Web Scraping:** Chrome WebDriver in headless mode  

---

## 🔧 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PythonDweeb/ClassmateAI
   cd classmate-ai
2. **Downloads the dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Run the application:**
   ```bash
   streamlit run main.py

---

## 🚀 How It Works

1. **User Input:** Students enter their school’s website into the app.
2. **Web Crawling:** ClassmateAI scrapes the website for relevant information and text content.
3. **Document Processing:** The scraped text is processed and split into manageable chunks.
4. **Vector Storage:** The processed text is converted into vectors using embeddings and stored in Weaviate for fast retrieval.
5. **Interactive Chat:** Students can ask questions, and ClassmateAI utilizes the stored data to provide accurate and timely responses. 🤖

---

## ⚙️ Libraries

- **Streamlit:** For the interactive UI  
- **Weaviate:** Vector-based search and storage  
- **Cerebras LLM:** Provides fast and accurate responses to user queries  
- **BeautifulSoup:** For web scraping   

---

## <250 LOC

With less than **250 lines of code**, ClassmateAI brings the power of web scraping, LLMs, and vector search into a simple and effective solution for students!

---

## 🏁 Ready to Get Started?

Just follow the setup instructions, enter a website, and begin exploring! 🎒

---
