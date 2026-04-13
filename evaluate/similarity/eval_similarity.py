import json
from cal_bert_score import calculate_bert_score
from cal_bleurt_score import calculate_bleurt_score
import numpy as np
from rouge_score import rouge_scorer
from nltk import sent_tokenize
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

def read(file_path):
    """
    :param file_path: path to jsonl file
    :return: list of data objects
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_object = json.loads(line)
            data.append(json_object)
    return data

def calculate(datas):
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    bertscore = calculate_bert_score(candidate, reference)
    bleurtscore = calculate_bleurt_score(candidate, reference)
    bertscore = np.array(bertscore)
    bleurtscore = np.array(bleurtscore)
    print("bertscore:", np.mean(bertscore), "bleurtscore:", np.mean(bleurtscore))
    return np.mean(bertscore), np.mean(bleurtscore)


def pre_rouge_processing(summary):
    summary = summary.replace("<n>", " ")
    summary = "\n".join(sent_tokenize(summary))
    return summary

def calue_rougscore(datas):
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    rouge1 = 0
    rouge2 = 0
    rougeL = 0
    rougeLsum = 0
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    length = len(candidate)
    rougeLsum_ls = []
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True, split_summaries=True)
    for i in tqdm(range(len(candidate)), desc='Evaluating rouge score', leave=False):
        scores = scorer.score(reference[i], pre_rouge_processing(candidate[i]))
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure
        rougeLsum += scores['rougeLsum'].fmeasure
        rougeLsum_ls.append(scores['rougeLsum'].fmeasure)

    return rouge1/length, rouge2/length, rougeL/length, rougeLsum/length


def calculate_bleu_score(datas):
    candidate = []
    reference = []
    for data in datas:
        candidate.append(data['predict'])
        reference.append(data['label'])
    bleu_sum = 0
    for i in range(len(candidate)):
        ref = [nltk.tokenize.word_tokenize(reference[i])]
        can = nltk.tokenize.word_tokenize(candidate[i])
        smooth = SmoothingFunction().method4
        score = sentence_bleu(ref, can, smoothing_function=smooth)
        bleu_sum += score
    return bleu_sum/len(candidate)

if __name__ == '__main__':

    
    datas_paths = [
        # please paste your absolute path to result.jsonl here, e.g. "/home/user/xxx/result.jsonl"
    ]

    results = []
    for datas_path in datas_paths:
        datas = read(datas_path)[:]
        rouge1, rouge2, rougeL, rougeLsum = calue_rougscore(datas)
        bertscroe, bleurtscore = calculate(datas)
        bleu = calculate_bleu_score(datas)
        result = (bertscroe, bleurtscore, bleu, rouge1, rouge2, rougeL, rougeLsum)
        results.append(result)
    
    for path, result in zip(datas_paths, results):
        print(f"Path: {path}")
        print(f"bertscore: {result[0]}, bleurtscore: {result[1]}, bleu: {result[2]}, rouge1: {result[3]}, rouge2: {result[4]}, rougeL: {result[5]}, rougeLsum: {result[6]}")