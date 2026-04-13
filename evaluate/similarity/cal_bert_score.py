# Calculate BERTScore between answer and reference output
import bert_score
import numpy as np
import torch
from tqdm import tqdm

def calculate_bert_score(candidates, references):
    """
    Calculate BERTScore between candidates and references
    :param candidates: list of candidate answers
    :param references: list of reference answers
    :return: F1 scores in [0, 1]
    """
    print("Evaluating bertscore")
    max_tokens = 500
    tokenizer = bert_score.utils.get_tokenizer(model_type="bert-base-multilingual-cased")
    for i in tqdm(range(len(candidates)), desc='Evaluating bertscore', leave=False):
        candidate = candidates[i]
        reference = references[i]
        candidate_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(candidate, add_special_tokens=True)))
        reference_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(reference, add_special_tokens=True)))
        if len(candidate_tokens) > max_tokens or len(reference_tokens) > max_tokens:
            candidate_tokens = candidate_tokens[:max_tokens]
            reference_tokens = reference_tokens[:max_tokens]
            candidate = tokenizer.convert_tokens_to_string(candidate_tokens)
            reference = tokenizer.convert_tokens_to_string(reference_tokens)
            min_len = min(len(candidate), len(reference))
            candidates[i] = candidate[:min_len]
            references[i] = reference[:min_len]

    P, R, F1 = bert_score.score(candidates, references, lang="en", verbose=True, model_type="bert-base-multilingual-cased", device='cuda:0', batch_size=16)
    return F1.numpy().tolist()