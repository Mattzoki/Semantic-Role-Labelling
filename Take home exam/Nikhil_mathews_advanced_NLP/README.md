Please excuse the format as this was created last minute using ChatGPT

# Challenging Semantic Role Labeling Models – SRL Evaluation Toolkit

This repository accompanies the take-home exam for the Advanced NLP course (2024–25). It includes code and data for evaluating two SRL models (Logistic Regression and BERT-based) on a manually constructed challenge dataset using CheckList-style tests.

---

## 📁 Repository Structure

```
.
├── challenge_dataset/                # JSON files per capability
│   ├── clefted_ARG0.json
│   ├── dative_ARG1.json
│   ├── dative_ARG2.json
│   ├── verbs_no_roles_ARG0.json
│   ├── passive_ARG1.json
│
├── combined_srl_challenge_dataset.json
├── final_data.csv                   # Model predictions with gold labels
├── srl_failure_rates_by_capability.csv
├── failure_rates.png
│
├── adv_utils.py                     # All utility functions (feature extraction, model inference, etc.)
├── capabilities.ipynb             # Script to run evaluation and generate outputs
├── requirements.txt
├── README.md
```

---

## 💾 Requirements

Install the dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🔧 Setup Instructions

2. **Download the two trained models:**

   - **Logistic Regression model** (includes model, vectorizer, and label encoder):  
     📥 [Download here](https://drive.google.com/your-logreg-model-link)  
     Place the following files in a folder named `models/logreg/`:

     - `logreg_model.pkl`
     - `vectorizer.pkl`
     - `label_encoder.pkl`

   - **BERT model (Fine-tuned SRL)**:  
     📥 [Download from HuggingFace](https://huggingface.co/your-bert-srl-model)  
     Place it under `models/bert/` or load directly from HuggingFace.

3. **Download Universal PropBank tokenizer:**  
   You may use `bert-base-cased` or the model-specific tokenizer:
   ```python
   from transformers import AutoTokenizer
   tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
   ```

---

## ▶️ Running the Evaluation

Use the following script to run predictions from both models and generate a CSV:

capabilities.ipynb

This will:

- Load challenge data from `combined_srl_challenge_dataset.json`
- Apply both models
- Save predictions to `final_data.csv`
- Compute failure rates (`srl_failure_rates_by_capability.csv`)
- Generate a bar plot (`failure_rates.png`)

---

## 📊 Outputs

- `final_data.csv`: Predictions by both models with gold labels
- `srl_failure_rates_by_capability.csv`: Failure rates per capability
- `failure_rates.png`: Visualized comparison of model failures

---

## 📘 Notes

- The challenge dataset is entirely hand-crafted for 5 capabilities.
- The evaluation uses failure rate (CheckList-style).
- Code is compatible with Python 3.9+ and tested on Jupyter + VSCode.

---

## ✍️ Author

- Matt — Vrije Universiteit Amsterdam — Text Mining Master's Track
