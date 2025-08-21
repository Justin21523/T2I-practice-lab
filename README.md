# 🎨 T2I-practice-lab

**T2I-practice-lab** is a structured practice and research repository focused on **Text-to-Image (T2I) generation models** and **LLM × RAG applications**.
The project integrates **PyTorch foundations**, **Transformer/LLM training**, **fine-tuning methods**, **retrieval-augmented generation (RAG)**, and **deployment pipelines** into a coherent learning lab.
It serves both as a **personal portfolio** and a **comprehensive training ground** for mastering modern AI systems.

---

## 📁 Project Structure

```bash
T2I-practice-lab/
├── README.md
├── .gitignore
├── .env.example               # Environment variables (HF_TOKEN, OPENAI_API_KEY)
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment setup
│
├── configs/
│   ├── model_configs.yaml
│   ├── cache_settings.yaml
│   └── evaluation_configs.yaml
│
├── notebooks/
│   ├── part_a_foundations/        # PyTorch & basic ML
│   ├── part_b_transformer_hf/     # Transformers + Hugging Face
│   ├── part_c_llm_applications/   # LLM apps (agents, evaluation, safety)
│   ├── part_d_finetuning/         # Fine-tuning (LoRA, QLoRA, DPO, RLHF)
│   ├── part_e_rag_agents/         # RAG + multi-agent collaboration
│   └── part_f_webui_api/          # WebUI & deployment
│
├── shared_utils/                  # Reusable Python utilities
├── tests/                         # Validation & smoke tests
└── docs/                          # Documentation (setup, roadmap, FAQ)
````

> See [`docs/learning_roadmap.md`](docs/learning_roadmap.md) for the **full 32+ notebook learning sequence**.

---

## ⚙️ Environment Setup

### Using Conda

```bash
conda env create -f environment.yml
conda activate t2i-lab
```

### Using pip

```bash
pip install -r requirements.txt
```

### Environment Variables

Copy `.env.example` → `.env` and set the following:

```bash
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_openai_key
```

---

## 📘 Learning Roadmap

The project is divided into **6 stages** (32+ notebooks):

* **Part A: Foundations** – PyTorch basics, CNNs, RNNs
* **Part B: Transformers & Hugging Face** – attention, datasets, model loading
* **Part C: LLM Applications** – function calling, reasoning, evaluation, agents
* **Part D: Fine-tuning** – LoRA, QLoRA, adapters, dataset curation, RLHF
* **Part E: RAG × Agents** – FAISS retrieval, multimodal RAG, multi-agent systems
* **Part F: WebUI & API** – Gradio UI, FastAPI + Docker deployment

🔥 **Priority Notebooks**:

* `nb13_function_calling_tools.ipynb`
* `nb26_rag_basic_faiss.ipynb`

---

## 🔧 Shared Utilities

* `cache_manager.py` – shared cache initialization
* `model_loader.py` – unified low-VRAM model loading
* `evaluation_utils.py` – evaluation metrics (ROUGE, BLEU, perplexity)
* `prompt_templates.py` – standardized prompt dictionary

---

## 🚀 Git Workflow

We follow a **feature-branch Git Flow**:

```bash
# Start development
git checkout -b develop

# New feature (e.g., notebook nb13)
git checkout -b feature/nb13-function-calling
# ... implement notebook ...
git commit -m "feat(notebooks): add nb13 function calling with LangChain tools"
git checkout develop
git merge --no-ff feature/nb13-function-calling

# Stage release
git checkout main
git merge --no-ff develop
git tag -a "v1.0-stage-c" -m "Complete LLM Applications stage"
```

See \[Git Workflow Strategy]\(docs/Git Workflow Strategy & Branching Model.md) for details.

---

## 📑 RAG × Prompt Dictionary

This project integrates **RAG pipelines** with a **Prompt/Style Dictionary**:

* **Retrieval layer**: FAISS, Chroma, Weaviate backends
* **Augmentation layer**: context injection (chunking, summarization, hybrid search)
* **LLM inference**: DeepSeek-R1, Qwen, LLaMA, GPT (optional APIs)
* **Prompt Dictionary**: centralized YAML/JSON templates for style, tone, safety

> Example: `prompt_templates.py` ensures consistent style across all agents (e.g., academic reasoning, step-by-step explanations, multilingual support).

---

## 📦 Release Strategy

* **Stage releases** aligned with notebook parts (A → F).
* **Semantic versioning**:

  * `v1.0` = Core LLM Applications complete
  * `v2.0` = Fine-tuning + RAG integration
  * `v3.0` = WebUI + Deployment

---

## 🤝 Contribution Guide

1. Fork the repo & create feature branch:

   ```bash
   git checkout -b feature/my-new-module
   ```
2. Add/modify notebooks or shared utilities.
3. Run tests in `/tests`.
4. Submit a Pull Request (PR).

---

## 📚 References

* Hugging Face 🤗 Transformers & Datasets
* PyTorch Lightning ⚡ for training loops
* FAISS, ChromaDB, Weaviate for retrieval
* Gradio, FastAPI, Docker for deployment

---

## 🏆 Goal

By completing **T2I-practice-lab**, you will:

* Gain **foundational deep learning skills**
* Build **LLM + RAG systems** end-to-end
* Deploy **AI-powered WebUI/API prototypes**
* Establish a **solid portfolio of practical AI projects**

---