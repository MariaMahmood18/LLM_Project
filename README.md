# LLM_Project
Improving Robustness of Radiology Report Summarization under OCR-style Text Noise using RAG
=======
# CS-818 Large Language Models - Semester Project

## Team Members
- Maria Mahmood - 537903
- Noor-ul-Ain Khalid - 537355

## Project Title
Improving Robustness of Radiology Report Summarization Under OCR-Style Text Noise Using RAG

## Track
Track A: Research-Oriented Project

## Project Overview
This project evaluates the robustness of LLM-based radiology report 
summarization under simulated OCR-style text noise. We investigate 
whether retrieval-augmented generation (RAG) can improve stability 
and factual consistency when input text contains character-level 
perturbations typical of OCR digitization errors.

## Dataset
Indiana University Chest X-Ray Reports (OpenI)
- 4000 radiology reports with XML structure
- Contains findings and impression sections
- Publicly available under CC BY-NC-ND 4.0 license
- URL: https://openi.nlm.nih.gov/


## Setup Instructions

### Prerequisites
- Python 3.10+
- Conda (recommended)
- Google Colab account (for compute)

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate cs818-project

# Install dependencies
pip install -r requirements.txt
```

## Reproduce Results

### Step 1 — Run the full pipeline
```bash
python scripts/run_pipeline.py --config configs/default.yaml
```

This will:
1. Load OpenI radiology reports (or synthetic fallback)
2. Inject OCR-style noise at configured level
3. Run clean baseline summarization (BART)
4. Run noisy baseline summarization
5. Build RAG index and run RAG-enhanced summarization
6. Print and save ROUGE-L + BERTScore results to `artifacts/logs/a2_results.json`

### Step 2 — Run tests
```bash
python -m pytest tests/
```

## Configuration

Edit `configs/default.yaml` to change:
- `data.max_samples`: Number of reports to use (reduce for speed)
- `noise.level`: OCR noise level (0.05, 0.10, 0.15, 0.20)
- `model.name`: HuggingFace model to use
- `rag.top_k`: Number of retrieved examples for RAG

---
