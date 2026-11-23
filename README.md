# Semantic Role Labelling

My work for the *Advanced NLP* course (MA Text Mining, Vrije Universiteit Amsterdam). Semantic role labelling (SRL) finds who did what to whom in a sentence. I built two SRL systems on the English EWT Universal Propositions data (CoNLL-U) and then stress-tested them with challenge sentences.

## Contents

### Assignment 1: Logistic regression SRL (`Assignment1/`)
- Converts CoNLL-U files to CoNLL, then expands each sentence into one copy per predicate.
- Extracts token and context features with spaCy (`en_core_web_md`) and trains a logistic regression classifier with `DictVectorizer`.
- Evaluates with classification reports and confusion matrices.
- Notebook: `Assignment1.ipynb`

### Assignment 2: BERT SRL (`Assignment2/`)
- Fine-tunes a BERT token classifier. The predicate is marked with a special `[PRED]` token added to the tokenizer.
- Settings: learning rate 2e-5, 3 epochs, weight decay 0.01, seed 42.
- Notebook: `Matt_BERT.ipynb`

### Take-home exam: challenging SRL models (`Take home exam/`)
- Behaviour-based testing in the spirit of CheckList: challenge sentences grouped by linguistic capability (for example passives with implied arguments, clefts and dative alternations).
- Reports a failure rate per capability for both models, saved in `srl_failure_rates_by_capability.csv` and `failure_rates.png`.
- Notebook: `capabilities.ipynb`, with helpers in `adv_utils.py` and the challenge set in `combined_srl_challenge_dataset.json`.
- The write-up is in [Main-Papers](https://github.com/Mattzoki/Main-Papers).

## Example finding

Failure rates differ a lot by capability. On passives with implied arguments, the logistic regression model failed on about 73% of items and BERT on none. Dative alternations were hard for both models.

## Running

```bash
pip install -r "Assignment1/Assignment 1/requirements.txt"
```

The trained models are too large for this repo. The notebooks link to download folders for the logistic regression and BERT models. Place them next to the notebook before running the exam notebook.

## Data

English Web Treebank with Universal Propositions: `en_ewt-up-{train,dev,test}.conllu` in `Assignment1/`.
