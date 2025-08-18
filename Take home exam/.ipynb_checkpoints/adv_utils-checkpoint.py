import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from spacy.tokens import Doc
import pickle
import matplotlib.pyplot as plt
import en_core_web_md
from tabulate import tabulate
import spacy
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, DataCollatorForTokenClassification, set_seed

from collections import Counter
from datasets import load_dataset
from evaluate import load
from datasets import Dataset
import numpy as np
import transformers
from transformers import AutoTokenizer

nlp = spacy.load("en_core_web_md")


def extract_features(df):
    """
    Extracts linguistic features for each token in a DataFrame grouped by sentence.

    Args:
        df (pandas.DataFrame): DataFrame with at least 'Sentence', 'Form', and 'Predicate' columns.

    Returns:
        list: A list of dictionaries, where each dictionary contains extracted features
              for a token in the dataset.
    """
    sentences = df.groupby("Sentence", sort=False)
    feature_sentences = []
    for sentence_number, group in sentences:
        for i, row in enumerate(group.to_dict(orient="records")):
            features = prepare_sentence(row, group, i)
            feature_sentences.append(features)
    return feature_sentences


def find_path(word, predicate):
    """
    Computes the dependency path from a word token to a predicate token within a dependency tree.

    Args:
        word (spacy.tokens.Token): The token from which the path starts.
        predicate (spacy.tokens.Token): The target predicate token.

    Returns:
        list: A list of dependency relations forming a path between the word and predicate.
              Direction is indicated using arrows (↑ for upward, ↓ for downward).
              Returns ['None'] if no valid path exists.
    """
    word_ancestors = {token: i for i, token in enumerate(word.ancestors)}
    predicate_ancestors = {token: i for i,
                           token in enumerate(predicate.ancestors)}

    if predicate.head == predicate:
        path = []
        node = word
        while node != predicate:
            if node == node.head:
                return ['None']
            path.append(node.dep_ + "↑")
            node = node.head
        path.append(predicate.dep_ + "*")
        return path

    if word.head == word:
        path = []
        node = predicate
        while node != word:
            if node == node.head:
                return ['None']
            path.append(node.dep_ + "↓")
            node = node.head
        path.append(word.dep_ + "*")
        return path

    common_ancestors = set(word_ancestors.keys()) & set(
        predicate_ancestors.keys())
    if not common_ancestors:
        return ['None']

    lca = min(common_ancestors,
              key=lambda token: word_ancestors[token] + predicate_ancestors[token])

    path = []
    node = word
    while node != lca:
        path.append(node.dep_ + "↑")
        node = node.head

    node = predicate
    down_path = []
    while node != lca:
        down_path.append(node.dep_ + "↓")
        node = node.head

    return path + [lca.dep_ + "*"] + list(reversed(down_path))


def prepare_sentence(row, group, idx):
    """
    Generates a feature dictionary for a given word in a sentence, based on its relationship 
    to the predicate. Extracted features include dependency path, named entity type, 
    lemma information, and part-of-speech context.

    Args:
        row (pandas.Series): Row containing information about the target word.
        group (pandas.DataFrame): DataFrame representing the full sentence.
        idx (int): Index of the current word in the sentence.

    Returns:
        dict: A dictionary of features capturing syntactic and semantic properties 
              of the word relative to the predicate.
    """
    sentence = group['Form'].tolist()
    predicate = group['Predicate'].tolist()
    word = row['Form']
    word_idx = idx
    pred_idx = predicate.index(1)

    if isinstance(sentence, list):
        spacy_doc = Doc(nlp.vocab, words=sentence)
        processed_doc = nlp(spacy_doc)
    else:
        processed_doc = nlp(sentence)

    doc_word = processed_doc[word_idx]
    doc_pred = processed_doc[pred_idx]

    path = find_path(doc_word, doc_pred)

    dir_dep = " ".join(path)
    lemma = doc_pred.lemma_
    depLemma = dir_dep + " " + lemma

    if doc_word.ent_type_:
        entLemma = doc_word.ent_type_ + " " + lemma
    else:
        entLemma = "O " + lemma

    contextPos = findContext(processed_doc, idx)
    PosDistLemma = " ".join([doc_word.pos_, str(word_idx - pred_idx), lemma])

    data = {
        "entLemma": entLemma,
        "depLemma": depLemma,
        "contextPos": contextPos
    }
    return data


def findContext(sent, i):
    """
    Extracts the part-of-speech (POS) context for a word in a sentence by combining 
    the POS tags of the previous, current, and next tokens.

    Args:
        sent (spacy.tokens.Doc): A SpaCy-parsed sentence.
        i (int): Index of the target token.

    Returns:
        str: A string in the format "PREV_CURR_NEXT" representing the POS context. 
             Uses '[START]' or '[END]' at sentence boundaries.
    """
    if i == 0:
        pre = '[START]'
    else:
        pre = sent[i-1].pos_

    current = sent[i].pos_

    if i == len(sent) - 1:
        next = '[END]'
    else:
        next = sent[i+1].pos_

    return pre + '_' + current + '_' + next


def find_nth_instance(lst, element, n):
    """
    Returns the index of the nth occurrence of a given element in a list.

    Args:
        lst (list): The list to search.
        element: The element to find.
        n (int): The occurrence number (0-based index).

    Returns:
        int: The index of the nth occurrence, or -1 if not found.
    """
    count = 0
    for i, val in enumerate(lst):
        if val == element:
            if count == n:
                return i
            count += 1
    return -1


def classify_data(model, vec, inputdata):
    """
    Transforms input features and generates predictions using the given model.

    Args:
        model: A trained classifier (e.g., logistic regression).
        vec: A vectorizer used to transform feature dictionaries.
        inputdata (list): A list of dictionaries containing extracted features.

    Returns:
        numpy.ndarray: Predicted labels for the input data.
    """
    features = vec.transform(inputdata)
    predictions = model.predict(features)
    return predictions


def stand_alone_sent_logreg(sent, predicate, log_model, log_vec, label_encoder):
    """
    Predicts semantic role labels for a sentence using a logistic regression model.

    Args:
        sent (list): The list of tokens in the sentence.
        predicate (list): A binary list marking predicate positions (1 for predicate, 0 otherwise).
        log_model: Trained logistic regression classifier.
        log_vec: DictVectorizer used for feature transformation.
        label_encoder: LabelEncoder used to decode predicted labels.

    Returns:
        list: Predicted semantic role labels for each token in the sentence.
    """
    predicate = [int(float(x)) for x in predicate]
    pred_num = predicate.count(1)
    all_tokens = []

    for i in range(pred_num):
        for idx, ele in enumerate(sent):
            token = {
                "Sentence": i,
                "Form": ele,
                "Predicate": 1 if find_nth_instance(predicate, 1, i) == idx else 0
            }
            all_tokens.append(token)

    data = pd.DataFrame(all_tokens)
    feats = extract_features(data)
    pred = classify_data(log_model, log_vec, feats)
    pred = list(label_encoder.inverse_transform(pred))

    return pred


def split_predicates_flat(sent, sent_target):
    """
    Duplicates a sentence for each predicate it contains, isolating one predicate at a time.

    Args:
        sent (list): List of tokens in the sentence.
        sent_target (list): Binary list marking predicates (1 for predicate, 0 otherwise).

    Returns:
        tuple: Two flattened lists:
            - sentence_duplicates: Sentences duplicated per predicate occurrence.
            - target_duplicates: Corresponding predicate masks, isolating one predicate at a time.
    """
    sentence_duplicates = []
    target_duplicates = []

    for idx, (word, target) in enumerate(zip(sent, sent_target)):
        if target == 1:
            sentence_copy = sent[:]
            target_copy = sent_target[:]

            for i in range(len(target_copy)):
                if i != idx:
                    target_copy[i] = 0

            sentence_duplicates.extend(sentence_copy)
            target_duplicates.extend(target_copy)

    return sentence_duplicates, target_duplicates


def main_preprocessor(sentence_data, BERTtokenizer):
    """
    Prepares sentence data for input into a BERT-based SRL model.

    Args:
        sentence_data (pd.DataFrame): A DataFrame with 'token' and 'chosen_predicate' columns,
                                      where 'chosen_predicate' marks the predicate(s) in the sentence.
        BERTtokenizer (PreTrainedTokenizer): Tokenizer used to encode the input for the BERT model.

    Returns:
        tuple: A tuple containing:
            - tokenized_data (list): List of dictionaries with 'input_ids', 'attention_mask',
              'word_ids', and 'tokenized_tokens' for each sentence.
            - final_dataset (Dataset): HuggingFace Dataset ready to be used for inference.
    """
    x = [{
        'tokens': sentence_data['token'].tolist(),
        'predicate': [0 if val == 0 else 1 for val in sentence_data['chosen_predicate']]
    }]

    y = []
    for sentence in x:
        tokens = sentence['tokens']
        predicates = sentence['predicate']
        processed_tokens = []
        processed_predicates = []
        for idx, (token, predicate) in enumerate(zip(tokens, predicates)):
            if predicate == 1:
                processed_predicates.append(predicates[idx])
                processed_tokens.append("[PRED]")
            processed_tokens.append(token)
            processed_predicates.append(predicates[idx])
        y.append({
            'tokens': processed_tokens,
            'predicates': processed_predicates
        })

    tokenized_data = []
    for sent in y:
        single_sent = {}
        tokenized_input = BERTtokenizer(
            sent["tokens"], is_split_into_words=True)
        tokens = BERTtokenizer.convert_ids_to_tokens(
            tokenized_input["input_ids"])
        word_ids = tokenized_input.word_ids()
        single_sent["input_ids"] = tokenized_input["input_ids"]
        single_sent["attention_mask"] = tokenized_input["attention_mask"]
        single_sent['word_ids'] = word_ids
        single_sent['tokenized_tokens'] = tokens
        tokenized_data.append(single_sent)

    processed_data = [{
        "input_ids": tokenized_data[0]["input_ids"],
        "attention_mask": tokenized_data[0]["attention_mask"]
    }]

    final_dataset = Dataset.from_dict({
        'input_ids': [example['input_ids'] for example in processed_data],
        'attention_mask': [example['attention_mask'] for example in processed_data]
    })

    return tokenized_data, final_dataset


def stand_alone_sent_bert(sent, target, BERTmodel, BERTtokenizer):
    """
    Predicts semantic role labels for a sentence using a fine-tuned BERT model.

    Args:
        sent (list of str): List of tokens in the sentence.
        target (list of int): Binary list indicating the predicate positions (1 = predicate, 0 = other).
        BERTmodel (PreTrainedModel): Trained BERT model for sequence labeling.
        BERTtokenizer (PreTrainedTokenizer): Tokenizer used with the BERT model.

    Returns:
        list of str: Predicted semantic role labels for each word in the original sentence.
    """
    target = [int(float(x)) for x in target]
    BERTtrainer = Trainer(model=BERTmodel, tokenizer=BERTtokenizer)
    sent, target = split_predicates_flat(sent, target)

    if len(sent) != len(target):
        raise ValueError(
            "The length of the sentence and the array to mark predicate must be the same.")

    label_to_id = {'ARG0': 45, 'ARG1': 0, 'ARG1-DSP': 1, 'ARG2': 17, 'ARG3': 20, 'ARG4': 16,
                   'ARG5': 26, 'ARGA': 3, 'ARGM-ADJ': 9, 'ARGM-ADV': 19, 'ARGM-CAU': 18, 'ARGM-COM': 24,
                   'ARGM-CXN': 5, 'ARGM-DIR': 21, 'ARGM-DIS': 15, 'ARGM-EXT': 46, 'ARGM-GOL': 8,
                   'ARGM-LOC': 47, 'ARGM-LVB': 44, 'ARGM-MNR': 4, 'ARGM-MOD': 11, 'ARGM-NEG': 14,
                   'ARGM-PRD': 10, 'ARGM-PRP': 23, 'ARGM-PRR': 13, 'ARGM-REC': 12, 'ARGM-TMP': 28,
                   'C-ARG0': 22, 'C-ARG1': 51, 'C-ARG1-DSP': 58, 'C-ARG2': 30, 'C-ARG3': 25, 'C-ARG4': 56,
                   'C-ARGM-ADV': 48, 'C-ARGM-COM': 35, 'C-ARGM-CXN': 2, 'C-ARGM-DIR': 29, 'C-ARGM-EXT': 52,
                   'C-ARGM-GOL': 57, 'C-ARGM-LOC': 54, 'C-ARGM-MNR': 40, 'C-ARGM-PRP': 39, 'C-ARGM-PRR': 32,
                   'C-ARGM-TMP': 31, 'C-V': 49, 'O': 27, 'R-ARG0': 43, 'R-ARG1': 55, 'R-ARG2': 33, 'R-ARG3': 37,
                   'R-ARG4': 50, 'R-ARGM-ADV': 34, 'R-ARGM-CAU': 36, 'R-ARGM-COM': 53, 'R-ARGM-DIR': 6,
                   'R-ARGM-GOL': 7, 'R-ARGM-LOC': 42, 'R-ARGM-MNR': 38, 'R-ARGM-TMP': 41}
    id_to_label = {v: k for k, v in label_to_id.items()}

    test_sentence = []
    for token, label in zip(sent, target):
        test_sentence.append(
            {"token": token, "chosen_predicate": token if label == 1 else 0})

    sentence_data = pd.DataFrame(test_sentence)
    tokenized_data, final = main_preprocessor(sentence_data, BERTtokenizer)

    predictions, _, _ = BERTtrainer.predict(final)
    predictions = np.argmax(predictions, axis=2)

    for idx, i in enumerate(tokenized_data):
        length = len(i['word_ids'])
        i['predictions'] = predictions[idx][:length]

    processed_predictions = []
    for data in tokenized_data:
        word_ids = data['word_ids']
        predictions = data['predictions']

        word_ids = [b for a, b in zip(data['tokenized_tokens'], word_ids) if a not in [
            '[CLS]', '[SEP]', '[PAD]', '[PRED]', '[UNK]', '[MASK]']]
        predictions = [b for a, b in zip(data['tokenized_tokens'], predictions) if a not in [
            '[CLS]', '[SEP]', '[PAD]', '[PRED]', '[UNK]', '[MASK]']]

        current_word_id = None
        current_predictions = []

        for idx, word_id in enumerate(word_ids):
            if word_id != current_word_id:
                if current_predictions:
                    predicted_label_id = Counter(
                        current_predictions).most_common(1)[0][0]
                    processed_predictions.append(
                        id_to_label[predicted_label_id])

                current_word_id = word_id
                current_predictions = [predictions[idx]]
            else:
                current_predictions.append(predictions[idx])

        if current_predictions:
            predicted_label_id = Counter(
                current_predictions).most_common(1)[0][0]
            processed_predictions.append(
                id_to_label[predicted_label_id])

    return processed_predictions
