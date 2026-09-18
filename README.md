# G.A.I.A. — Global Agroecology Intelligent Assistant
> **Darukaa.Earth AI Environmentalist Chatbot Submission**

G.A.I.A. is an AI-powered Nature Intelligence Agent designed to evaluate ecosystem health, diagnose soil/land degradation, and generate evidence-backed, multi-variable restoration plans.

---

## 🏗️ System Architecture & Workflow

The platform operates on a Retrieval-Augmented Generation (RAG) architecture built specifically to prevent generic responses and enforce scientific grounding.

1. **Frontend Layer (`app1.py`):** Interactive Streamlit interface supporting free-form text input, structured metric forms, and coordinate spatial context.
2. **Knowledge Retrieval Layer (`giaa_engine.py`):** Converts queries into vector embeddings using `BAAI/bge-small-en-v1.5` to retrieve peer-reviewed scientific context from ChromaDB.
3. **Reasoning Engine (`Groq API`):** Employs `qwen/qwen3.8-27b` with system prompt guardrails enforcing multi-variable analysis and evidence citations.

---

## 🗄️ Database & Schema Design

* **Vector Database:** `ChromaDB` persistent local storage (`./chroma_db`).
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors).
* **Ingested Corpus:** Peer-reviewed FAO soil reports, IPCC climate assessment chapters, and global agroforestry datasets chunked via LlamaIndex `SentenceSplitter` (chunk size: 512, overlap: 50).
* **Retrieval Schema:** Vector similarity search returning top-k relevant document chunks along with metadata citations.

---

## 🧪 Multi-Metric & Biophysical Reasoning

G.A.I.A. evaluates interconnected environmental metrics rather than isolated variables:
* **Soil Health:** pH, Soil Organic Carbon (SOC %), moisture levels.
* **Climate Factors:** Annual rainfall, temperature variance, seasonal droughts.
* **Land & Habitat:** Land use cover, degradation levels, biodiversity indicators.

Every generated recommendation delivers:
1. **Actionable Intervention**
2. **Scientific Mechanism & Biophysical Reasoning**
3. **Quantified Impacted Metrics & Estimated Gains**
4. **Time Horizon (Short, Medium, Long-Term)**
5. **Direct Source Citations**

---

## ⚙️ Local Setup Instructions

### Prerequisites
* Python 3.10 or higher
* Groq API Key

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/gaia-bot.git](https://github.com/YOUR_GITHUB_USERNAME/gaia-bot.git)
   cd gaia-bot