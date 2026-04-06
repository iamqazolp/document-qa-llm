# Document QA LLM (RAG Agent API)

A professional API-driven question-answering system based on personal documents, utilizing the **Retrieval-Augmented Generation (RAG)** architecture. Built with **FastAPI**, this project supports high-speed cloud inference via **Groq** and private local inference using **Ollama**.

---

## Key Features

- **FastAPI Backend:** High-performance framework with automatic interactive API documentation (Swagger UI).
- **Hybrid LLM Support:**
  - **Groq Integration:** Ultra-fast cloud-based inference.
  - **Ollama Integration:** Privacy-focused local model execution.
- **Vector Search:** Efficient document retrieval powered by **ChromaDB**.
- **Deployment Ready:** Fully containerized with Docker for seamless environment parity.

---

## Technologies Used

- **Backend:** Python, FastAPI, Uvicorn
- **LLM Cloud:** Groq API
- **LLM Local:** Ollama
- **Vector Database:** ChromaDB
- **Deployment:** Docker

---

## Getting Started

### 1. Clone the repository

```bash
git clone [https://github.com/iamqazolp/document-qa-llm.git](https://github.com/iamqazolp/document-qa-llm.git)
cd document-qa-llm
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration (Environment Variables)

Create a file named `.env` in the root directory. This project requires several environment variables to function correctly. Ensure you define the following:

```env
COLLECTION_NAME = ""
EMBEDDING_MODEL = "" #all-MiniLM-L6-v2 is recommended
DATABASE_PATH = ""
DOCUMENTS_FOLDER = ""
GROQ_API_KEY = ""
LOCAL_MODEL_NAME = "" #llama3.2:3b is recommended since RAG reduces the need of reasoning
CLOUD_MODEL_NAME = ""
```

### 5. Build the Vector Database

Before running the API, you must process your documents and initialize the vector store.

1. Place your documents (currently only `.pdf` and `.txt` files are supported) into the `documents/` folder.
2. Run the database builder script:

   ```bash
   python db_builder.py
   ```

   - This step uses **ChromaDB** to create embeddings and store them locally based on your `.env` configuration.

---

## Running the Application

To start the FastAPI server with live-reloading:

```bash
uvicorn app:app --reload
```

Once started, navigate to the following URL to test the API endpoints using the interactive Swagger UI:
**http://127.0.0.1:8000/docs**
<img width="2729" height="1533" alt="image" src="https://github.com/user-attachments/assets/a4f05305-3acc-40cf-b335-6e2e6dbeb12d" />

---

## Running with Docker

You can run the entire stack within a Docker container. Both cloud (Groq) and local (Ollama) modes are supported.

> **Note:** For local mode, Ollama must be running on your host machine with the model already pulled (`ollama pull llama3.2:3b`). Inference speed in local mode depends on your machine's hardware.

**Step 1: Build and start the container**

```bash
docker compose up --build
```

**Step 2: Switch between modes (optional)**

Cloud mode (Groq) is the default. To switch to local Ollama:

```bash
curl -X PUT "http://localhost:8000/change_mode?mode=local"
```

Access the API documentation at: **http://127.0.0.1:8000/docs**

---

## Contributing

Contributions to improve RAG performance, expand features, or fix bugs are welcome. Please open an **Issue** or submit a **Pull Request**.

---

_Developed by [iamqazolp](https://github.com/iamqazolp)_
