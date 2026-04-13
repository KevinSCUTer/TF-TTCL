import re
import json
import inspect
from typing import Optional, List
from functools import partial
from dataclasses import dataclass
from enum import Enum

class MatchReason(Enum):
    STRING_EXACT = "string_exact"
    NUMERIC_EQUAL = "numeric_equal"
    SYMBOLIC_EQUAL = "symbolic_equal"
    LATEX_NORMALIZED = "latex_normalized"
    
    STRING_MISMATCH = "string_mismatch"
    NUMERIC_MISMATCH = "numeric_mismatch"
    SYMBOLIC_MISMATCH = "symbolic_mismatch"
    NO_ANSWER_FOUND = "no_answer_found"
    EXTRACTION_ERROR = "extraction_error"
    COMPARISON_ERROR = "comparison_error"
    UNKNOWN = "unknown"

@dataclass
class EvalResult:
    idx: int
    is_correct: bool
    ground_truth: str
    ground_truth_normalized: str
    extracted_answer: Optional[str]
    extracted_normalized: Optional[str]
    match_reason: MatchReason
    match_detail: str
    question: str
    response: str
    extraction_trace: List[str]

def remove_boxed(s):
    if s is None:
        return None
    
    start = s.find("{")
    end = s.rfind("}")
    
    if start != -1 and end != -1 and end > start:
        return s[start+1:end]
    
    return None

def extract_last_candidates(text: str, max_candidates: int = 1) -> List[str]:
    if not text:
        return []
    
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    candidates = []
    all_matches = []
    
    for match in re.finditer(r'\$([^\$]+)\$', text):
        all_matches.append((match.start(), match.group(1)))
    
    for match in re.finditer(r'\\\[(.*?)\\\]', text):
        all_matches.append((match.start(), match.group(1)))

    num_pattern = r'-?\d*\.?\d+(?:[eE][-+]?\d+)?'
    for match in re.finditer(num_pattern, text):
        val = match.group()
        if "." in val or "e" in val.lower() or len(val) < 5:
             all_matches.append((match.start(), val))
    
    for match in re.finditer(r'answer is\s*(.*)', text, re.IGNORECASE):
        val = match.group(1).strip().split('\n')[0]
        if val:
            all_matches.append((match.start(), val))

    all_matches.sort(key=lambda x: x[0], reverse=True)
    
    seen = set()
    for _, val in all_matches:
        val_clean = val.strip().strip('.').strip()
        if val_clean and val_clean not in seen:
            candidates.append(val_clean)
            seen.add(val_clean)
            if len(candidates) >= max_candidates:
                break
    
    return candidates

def extract_last_5_candidates(text: str) -> List[str]:
    return extract_last_candidates(text, max_candidates=5)

def process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ') if len(completion.split('The answer is: '))>1 else completion.split('The answer is ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        print(f"extract_ans: {extract_ans}")
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False


def last_boxed_only_string(string):
    if string is None:
        return None
    
    string = string.replace("\x08oxed", "\\boxed")
    string = string.replace("\x0cfbox", "\\fbox")
    
    string = string.replace("/boxed", "\\boxed")
    string = string.replace("/fbox", "\\fbox")
    
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            idx = string.rfind("boxed{")
            if idx < 0:
                idx = string.rfind("fbox{")
                if idx < 0:
                    return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    # except AssertionError:
    except Exception:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    if string is None:
        return ""
    
    string = string.replace("\x08oxed", "\\boxed")
    string = string.replace("\x0cfbox", "\\fbox")
    string = string.replace("\x0crac", "\\frac")
    
    string = string.replace("\n", "")

    string = string.replace("\\!", "")

    string = string.replace("\\\\", "\\")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = fix_sqrt(string)

    string = string.replace(" ", "")

    string = fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    string = fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False, numeric_tol=0.0):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        if ss1 == ss2:
            return True
        if numeric_tol > 0:
            try:
                v1 = float(ss1.replace(',', ''))
                v2 = float(ss2.replace(',', ''))
                if abs(v2) > 1e-9:
                    if abs(v1 - v2) / abs(v2) <= numeric_tol:
                        return True
                else:
                    if abs(v1 - v2) <= numeric_tol:
                        return True
            except (ValueError, TypeError):
                pass
        return False
    except Exception:
        return str1 == str2

LEGACY_NUMERIC_TOL = 0.01

def judge_legacy(completion, answer, use_last_number=True, verbose=False):
    if "\\boxed" in str(answer):
        extracted_label = extract_answer_between_boxed(str(answer), use_last_number=False)
        if extracted_label:
            answer = extracted_label
    answer = str(answer).replace("<think>\n\n</think>", "").strip("\n")

    def _match(pred, label):
        if pred is None or label is None:
            return False
        if is_equiv(pred, label, verbose=verbose, numeric_tol=LEGACY_NUMERIC_TOL):
            return True
        try:
            pred_s = strip_string(pred)
            label_s = strip_string(label)
            if "\\times" in pred_s or "\\times" in label_s or "e" in pred_s.lower() or "e" in label_s.lower():
                pred_norm = normalize_scientific_str(pred_s)
                label_norm = normalize_scientific_str(label_s)
                p_val = float(pred_norm)
                l_val = float(label_norm)
                if abs(l_val) > 1e-9 and abs(p_val - l_val) / abs(l_val) <= LEGACY_NUMERIC_TOL:
                    return True
        except Exception:
            pass
        return False

    extract_ans = extract_answer_between_boxed(completion, use_last_number=True)
    if extract_ans and _match(extract_ans, answer):
        return True, extract_ans

    candidates = extract_last_candidates(completion, max_candidates=2)
    for cand in candidates:
        if _match(cand, answer):
            if verbose:
                print(f"Legacy matched candidate: {cand}")
            return True, cand

    return False, extract_ans or (candidates[0] if candidates else None)

def extract_gsm8k_answer_number_robust(completion, label, max_candidates=1):
    
    tokens = re.split(r'[\s\n\n]+', completion)
    
    tokens_with_numbers = [
        token for token in tokens
        if re.search(r'-?\d', token)
    ]
    
    cleaned_numbers = []
    for token in tokens_with_numbers:
        cleaned = re.sub(r'[^\d,\.-]', '', token)
        cleaned = cleaned.replace(',', '').strip('.')
        
        if (cleaned and
            cleaned.count('.') <= 1 and
            cleaned.count('-') <= 1 and
            re.match(r'^-?\d*\.?\d+$', cleaned)):
            try:
                val = float(cleaned)
                if val.is_integer():
                    normalized = str(int(val))
                else:
                    normalized = str(val)
                cleaned_numbers.append(normalized)
            except:
                continue
    
    if not cleaned_numbers:
        return None
    
    candidates = []
    seen = set()
    
    for num in reversed(cleaned_numbers):
        if num not in seen:
            candidates.append(num)
            seen.add(num)
            if len(candidates) >= max_candidates:
                break
    
    try:
        label_val = float(str(label).replace(',', '').strip())
        if label_val.is_integer():
            label_normalized = str(int(label_val))
        else:
            label_normalized = str(label_val)
    except:
        label_normalized = str(label).strip()
    
    for candidate in candidates:
        if candidate == label_normalized:
            return candidate
    
    return candidates[0] if candidates else None
    
def extract_gsm8k_answer_number(completion):
    tokens = re.split(r'[\s\n\n]+', completion)
    tokens_with_numbers = [
        token for token in tokens
        if re.search(r'-?\d', token)
    ]
    cleaned_numbers = [re.sub(r'[^\d,\.-]', '', token) for token in tokens_with_numbers]
    
    if cleaned_numbers:
        extracted_number = cleaned_numbers[-1].replace(',', '').strip('.')
        if extracted_number.count('.') > 1 or extracted_number.count('-') > 1 or not re.match(r'^-?\d*\.?\d*$', extracted_number):
            return None
        try:
            return str(round(float(extracted_number))) 
        except:
            print(f"cannot convert to float: {extracted_number}")
            return None

    return  None


def extract_answer_between_boxed(completion, use_last_number=False):
    extract_ans = None
    extract_ans = remove_boxed(last_boxed_only_string(completion))
    if not extract_ans and use_last_number:
        pattern = "-?\d*\.?\d+"
        pred = re.findall(pattern, completion.replace(",", ""))
        if len(pred) >= 1:
            extract_ans = pred[-1]
    return extract_ans



def process_minerva_results(completion, answer, use_last_number=False, verbose=False, max_candidates=1):
    actual_use_last = use_last_number if max_candidates > 0 else False
    extract_ans = extract_answer_between_boxed(completion, use_last_number=actual_use_last)
    
    if "\\boxed" in answer:
        extracted_label = extract_answer_between_boxed(answer, use_last_number=False)
        if extracted_label:
            answer = extracted_label

    answer = answer.replace("<think>\n\n</think>", "").strip("\n")
    
    def check_match(pred, label):
        if pred is None or label is None: return False
        try:
            pred_s = strip_string(pred)
            label_s = strip_string(label)
            
            if "\\times" in pred_s or "e" in pred_s or "\\times" in label_s or "e" in label_s:
                try:
                    pred_norm = normalize_scientific_str(pred_s)
                    label_norm = normalize_scientific_str(label_s)
                    p_val = float(pred_norm)
                    l_val = float(label_norm)
                    if abs(p_val - l_val) / (abs(l_val) + 1e-9) < 0.05:
                        return True
                except:
                    pass
            
            try:
                p_val = float(pred_s)
                l_val = float(label_s)
                if abs(p_val - l_val) / (abs(l_val) + 1e-9) < 0.05:
                    return True
            except:
                pass

            return pred_s == label_s or is_equiv(pred, label)
        except Exception:
            return is_equiv(pred, label)

    if extract_ans and check_match(extract_ans, answer):
        return True, extract_ans
        
    if max_candidates > 0:
        candidates = extract_last_candidates(completion, max_candidates=max_candidates)
        for cand in candidates:
            if check_match(cand, answer):
                if verbose: print(f"Matched candidate: {cand}")
                return True, cand
            
    return False, extract_ans if extract_ans else (candidates[0] if (max_candidates > 0 and 'candidates' in locals() and candidates) else None)
import math
def normalize_scientific_str(s):
        s = re.sub(r'\s*\\times\s*10\^\{([^}]*)\}', lambda m: f'e{m.group(1)}', s)
        s = re.sub(r'\s*x\s*10\^', 'e', s)
        s = re.sub(r'([0-9])\s*x\s*', r'\1e', s)
        s = s.replace('^', '**')

        try:
            value = float(eval(s))
        except Exception:
            raise ValueError(f"无法解析字符串: {s}")

        if value == 0:
            return "0"
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = round(value / (10 ** exponent), 1)
        
        if mantissa == int(mantissa):
            mantissa = int(mantissa)

        return f"{mantissa}e{exponent}"

def process_math_aime_results(completion, answer, use_last_number=False, verbose=False, max_candidates=1):
    actual_use_last = use_last_number if max_candidates > 0 else False
    extract_ans = extract_answer_between_boxed(completion, use_last_number=actual_use_last)
    
    if "\\boxed" in answer:
        extracted_label = extract_answer_between_boxed(answer, use_last_number=False)
        if extracted_label:
            answer = extracted_label

    answer = answer.replace("<think>\n\n</think>", "").strip("\n")
    
    if extract_ans and is_equiv(extract_ans, answer, verbose=verbose):
        return True, extract_ans
        
    if max_candidates > 0:
        candidates = extract_last_candidates(completion, max_candidates=max_candidates)
        for cand in candidates:
            if is_equiv(cand, answer, verbose=verbose):
                if verbose: print(f"Matched candidate: {cand}")
                return True, cand
            
    return False, extract_ans if extract_ans else (candidates[0] if (max_candidates > 0 and 'candidates' in locals() and candidates) else None)


def fraction_to_decimal(fraction_str):
    match = re.match(r'\\frac\{(\d+)\}\{(\d+)\}', fraction_str)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    
    try:
        result = numerator / denominator
        return result
    except ZeroDivisionError:
        return None

def process_gsm8k_results(completion, answer, use_last_number=False, verbose=False, max_candidates=1):
    answer = str(answer).replace(',', '')
    actual_use_last = use_last_number if max_candidates > 0 else False
    extract_ans = extract_answer_between_boxed(completion, use_last_number=actual_use_last)
    if extract_ans:
        if is_equiv(extract_ans, answer):
            return True, extract_ans
        try:
            if float(str(extract_ans).replace(',', '')) == float(str(answer).replace(',', '')):
                return True, extract_ans
        except:
            pass
            
    if max_candidates > 0:
        extract_ans_robust = extract_gsm8k_answer_number_robust(completion, answer, max_candidates=max_candidates)
        
        if verbose:
                print(f"extract_ans: {extract_ans_robust}")
                
        if extract_ans_robust:
            if is_equiv(extract_ans_robust, answer):
                return True, extract_ans_robust
            try:
                if float(str(extract_ans_robust).replace(',', '')) == float(str(answer).replace(',', '')):
                    return True, extract_ans_robust
            except:
                pass
                
            return False, extract_ans if extract_ans else extract_ans_robust
    
    return False, extract_ans

def process_gsm8k_results_v2(completion, answer, use_last_number=False, verbose=False, max_candidates=1):
    answer = str(answer).replace(',', '')
    actual_use_last = use_last_number if max_candidates > 0 else False
    extract_ans = extract_answer_between_boxed(completion, use_last_number=actual_use_last)
    if extract_ans:
        extract_ans_stripped, answer_stripped = strip_string(extract_ans), strip_string(answer)
        if verbose:
            print(extract_ans_stripped, answer_stripped)
        try:
            if str(round(float(extract_ans_stripped.replace(",", "")))) == answer_stripped.replace(",", ""):
                return True, extract_ans
            else:
                return False, extract_ans
        except:
            return False, extract_ans
    else:
        return False, None


def is_braced(s: str, left="({", right=")}"):
    flag = True
    for l, r in zip(left, right):
        f = s.count(l) == s.count(r)
        flag = flag and f
    return flag

def split_equal(content: str):
    results = []
    if content.count("=") == 0:
        results.append(content.strip())

    elif content.count("=") == 1:
        idx = content.find("=")
        if is_braced(content[:idx]):
            results.append(content.split("=")[-1].strip())
        else:
            results.append(content.strip())

    elif content.count("=") > 1:
        idx_list = [i for i, char in enumerate(content) if char == '=']
        if "," in content[idx_list[0]:idx_list[1]]:
            for i, idx in enumerate(idx_list):
                if i+1 < len(idx_list):
                    c = content[idx_list[i]+1:idx_list[i+1]]
                    results.append(c.split(",")[0])
                else:
                    results.append(content[idx+1:])
        else:
            results.append(content)
    
    return ', '.join(results)

def strip_text(text: str):
    text = text.lower().replace(" ", "")
    text = text.replace("dfrac", "frac")

    pattern = 'text\{(.*?)\}'
    if match := re.search(pattern, text):
        text = match.group(1)
    return text

def process_batch(completion_batch: List[str], answer_batch: List[str], process_fn, verbose=False):
    results = []
    for completion, answer in zip(completion_batch, answer_batch):
        res = process_fn(completion, answer, use_last_number=True, verbose=verbose)
        if isinstance(res, bool):
            results.append((res, None))
        else:
            results.append(res)
    return results

def auto_verify(completion_list: List[str], answer_list: List[str], dataset_type: str, verbose: bool = False):
    if dataset_type == "gsm8k1":
        process_fn = process_gsm8k_results
    elif dataset_type == "gsm8k2":
        process_fn = process_gsm8k_results_v2
    elif dataset_type in ["math_500", "aime"]:
        process_fn = process_math_aime_results
    elif dataset_type == "minerva":
        process_fn = process_minerva_results
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")    
    
    return process_batch(completion_list, answer_list, process_fn, verbose)

def compare_answers_detailed(
    ground_truth: str,
    extracted: Optional[str],
    dataset_type: str
):
    if extracted is None:
        return False, MatchReason.NO_ANSWER_FOUND, "No answer extracted from response"
    
    if str(ground_truth).strip() == str(extracted).strip():
        return True, MatchReason.STRING_EXACT, f"Exact match: '{ground_truth}'"
    
    try:
        if is_equiv(extracted, ground_truth):
             try:
                 float(extracted)
                 float(ground_truth)
                 return True, MatchReason.NUMERIC_EQUAL, f"Numeric equal: {extracted} ~ {ground_truth}"
             except:
                 return True, MatchReason.SYMBOLIC_EQUAL, f"Symbolic equal: {extracted} ~ {ground_truth}"
    except Exception as e:
        return False, MatchReason.COMPARISON_ERROR, f"Error in is_equiv: {e}"

    return False, MatchReason.SYMBOLIC_MISMATCH, f"Mismatch: {extracted} != {ground_truth}"

def verify_single_detailed(
    idx: int,
    pred: str,
    label: str,
    question: str,
    dataset_type: str,
    process_fn,
    verbose: bool = False,
    max_candidates: int = 1
) -> EvalResult:
    try:
        sig = inspect.signature(process_fn)
        actual_use_last = True if max_candidates > 0 else False
        
        if 'max_candidates' in sig.parameters:
            is_correct, extracted = process_fn(pred, label, use_last_number=actual_use_last, verbose=verbose, max_candidates=max_candidates)
        else:
            is_correct, extracted = process_fn(pred, label, use_last_number=actual_use_last, verbose=verbose)
    except Exception as e:
        return EvalResult(
            idx=idx,
            is_correct=False,
            ground_truth=str(label),
            ground_truth_normalized="",
            extracted_answer=None,
            extracted_normalized=None,
            match_reason=MatchReason.EXTRACTION_ERROR,
            match_detail=f"Error in process_fn: {e}",
            question=question,
            response=pred,
            extraction_trace=[]
        )

    match_reason = MatchReason.UNKNOWN
    match_detail = ""
    
    if is_correct:
        _, reason, detail = compare_answers_detailed(label, extracted, dataset_type)
        match_reason = reason
        match_detail = detail
        if match_reason in [MatchReason.SYMBOLIC_MISMATCH, MatchReason.NO_ANSWER_FOUND, MatchReason.NUMERIC_MISMATCH]:
             match_reason = MatchReason.SYMBOLIC_EQUAL 
             match_detail = "Matched by process_fn logic (e.g. candidates)"
    else:
        if extracted is None:
            match_reason = MatchReason.NO_ANSWER_FOUND
            match_detail = "No answer extracted"
        else:
            match_reason = MatchReason.SYMBOLIC_MISMATCH
            match_detail = f"Mismatch: {extracted} != {label}"

    return EvalResult(
        idx=idx,
        is_correct=is_correct,
        ground_truth=str(label),
        ground_truth_normalized=str(label),
        extracted_answer=str(extracted) if extracted is not None else None,
        extracted_normalized=str(extracted) if extracted is not None else None,
        match_reason=match_reason,
        match_detail=match_detail,
        question=question,
        response=pred,
        extraction_trace=[]
    )

def auto_verify_detailed(
    preds: List[str],
    labels: List[str],
    questions: List[str],
    dataset_type: str,
    verbose: bool = False,
    max_candidates: int = 1,
    equiv_mode: str = None
) -> List[EvalResult]:
    if equiv_mode == "legacy" and dataset_type not in ["gsm8k1", "gsm8k2", "mawps"]:
        process_fn = judge_legacy
        max_candidates = 2
    elif dataset_type == "gsm8k1":
        process_fn = process_gsm8k_results
    elif dataset_type == "gsm8k2":
        process_fn = process_gsm8k_results_v2
    elif dataset_type in ["math_500", "aime"]:
        process_fn = process_math_aime_results
    elif dataset_type == "minerva":
        process_fn = process_minerva_results
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    results = []
    for i, (pred, label, question) in enumerate(zip(preds, labels, questions)):
        result = verify_single_detailed(
            idx=i+1,
            pred=pred,
            label=label,
            question=question,
            dataset_type=dataset_type,
            process_fn=process_fn,
            verbose=verbose,
            max_candidates=max_candidates
        )
        results.append(result)
    
    return results