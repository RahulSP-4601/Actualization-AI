# Actualization-AI

This project is an **assessment** where the goal was to build a pipeline that:

1. **Parses contracts (PDFs)** into structured JSON with sections and clauses.
2. **Post-processes the parsed JSON using an LLM** (OpenAI API) for cleanup and refinement.

---

## 🚀 Features

- Extracts **title, contract type, effective date, sections, and clauses** from PDF contracts.
- Works with both **short contracts** (2–3 pages) and **large contracts** (30+ pages).
- Integrated with **OpenAI GPT models** for post-processing.
- Clean CLI-based workflow to run end-to-end.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/RahulSP-4601/Actualization-AI.git
cd Actualization-AI
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Mac/Linux
.venv\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

_(If `requirements.txt` is missing, install manually:)_

```bash
pip install pdfplumber pillow pytesseract dateparser python-dotenv openai
```

### 4. Add `.env` File

Create a `.env` file in the project root with your OpenAI key:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## ▶️ How to Run

### Step 1 – Parse PDF (No LLM Integration)

```bash
python rahul_panchal.py input2.pdf output.json
```

This creates a structured JSON (`output.json`) from the contract.

### Step 2 – Post-process with LLM Integration

```bash
python run_pipeline.py input1.pdf final_output.json
```

This will:

1. Parse the PDF (`rahul_panchal.py`)
2. Send output to OpenAI for refinement (`llm_post.py`)
3. Save the final result in `final_output.json`

---

## 📂 File Overview

- **rahul_panchal.py** → Contract parsing (PDF → JSON).
- **llm_post.py** → Post-processing using LLM (OpenAI).
- **run_pipeline.py** → Full pipeline runner (PDF → JSON → Refined JSON).
- **input1.pdf / input2.pdf** → Sample test contracts.
- **output.json / output1.json** → Example parsed results.
- **.gitignore** → Ensures `.env` and other sensitive files are ignored.

---

## 📌 Example

```bash
python run_pipeline.py input2.pdf output.json
```

Output:

```json
{
  "title": "Sample Contract",
  "contract_type": "Agreement",
  "effective_date": "2024-01-01",
  "sections": [...]
}
```

---

## 🔗 GitHub

Project Link: [Actualization-AI](https://github.com/RahulSP-4601/Actualization-AI.git)

---
