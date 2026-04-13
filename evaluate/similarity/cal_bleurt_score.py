from bleurt import score
from tqdm import tqdm
def calculate_bleurt_score(candidate, reference):
    """
    Calculate BLEURT score between candidate and reference
    :param candidate: list of candidates
    :param reference: list of references
    :return: BLEURT scores in [0, 1]
    """
    results = []
    # your checkpoint path here
    checkpoint = "/path/to/bleurt/bleurt-base-128"
    scorer = score.BleurtScorer(checkpoint)
    for i in tqdm(range(len(candidate)), desc='Evaluating bleurtscore', leave=False):
        scores = scorer.score(references=[reference[i]], candidates=[candidate[i]])
        assert isinstance(scores, list) and len(scores) == 1
        results.append(scores[0])
    return results