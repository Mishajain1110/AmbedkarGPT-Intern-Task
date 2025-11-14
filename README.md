# Retrieval-Based QA with Ollama and Chroma

This project uses Ollama LLM and Chroma vector database to perform retrieval-based question answering on a text file (`speech.txt`). It splits the text into chunks, creates embeddings, and uses the LLM to answer queries based on the content.

---

## 1. Create a Python Virtual Environment

### Windows:

```bash
python -m venv venv
venv\Scripts\activate
````

### Linux / macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 2. Install Ollama

Follow instructions from [Ollama official site](https://ollama.com/) for your OS.

### Windows:

* Download the installer and follow the setup steps.

### Linux:

```bash
# Example installation (check Ollama docs for exact commands)
curl -fsSL https://ollama.com/install.sh | bash
```

---

## 3. Pull the Mistral Model

```bash
ollama pull mistral
```

This downloads the `mistral` model to your local Ollama installation.

---

## 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Prepare Your Text File

* Place your `speech.txt` (or `input.txt`) in the **same directory** as your main Python script (`main.py`).

---

## 6. Run the Main Script

```bash
python main.py
```

**Example query in the script:**

```python
query = "What are the main points or key ideas?"
```

**Example output:**

```
Response by the LLM: [The main points from your speech]
```

---
