# python -m eval.cal_acc
from acc_utils import auto_verify, auto_verify_detailed
from log_writer import write_detailed_csv, write_detailed_log
from typing import List, Union, Dict, Optional, Tuple, Any
import os
import json
import pandas as pd
import numpy as np

# Please modify to absolute paths if you want to use the ground truth labels for GSM8K, MATH-500, Minerva, AIME24 datasets. 
GSM8K_PARQUET_PATH = "path/to/data/MATH/reference/gsm8k/main/test-00000-of-00001.parquet"
MATH500_JSONL_PATH = "path/to/data/MATH/reference/MATH-500/test.jsonl"
MINERVA_JSONL_PATH = "path/to/data/MATH/reference/minerva/test.jsonl"
AIME_JSONL_PATH = "path/to/data/MATH/reference/aime24/aime.jsonl"

GOLD_PATCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patches")
GOLD_PATCH_MAP = {
    "math_500": "human_eval_math500.json",
}

def _load_gold_patches(dataset_type):
    filename = GOLD_PATCH_MAP.get(dataset_type)
    if not filename:
        return {}
    patch_file = os.path.join(GOLD_PATCH_DIR, filename)
    if not os.path.exists(patch_file):
        return {}
    with open(patch_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get(dataset_type, {})

def _apply_gold_patches(labels, patches):
    if not patches:
        return labels, []
    patched_labels = list(labels)
    patched_info = []
    for str_id, patch in patches.items():
        idx = int(str_id) - 1
        if 0 <= idx < len(patched_labels):
            old_label = patched_labels[idx]
            patched_labels[idx] = patch["correct_answer"]
            patched_info.append((int(str_id), old_label, patch["correct_answer"]))
    return patched_labels, patched_info

def _parse_jsonl_line(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        return json.loads(line)
    except Exception:
        try:
            return eval(line)
        except Exception:
            return {}


def _read_jsonl_objects(jsonl_path: str) -> List[Dict[str, Any]]:
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        text = f.read()

    records: List[Dict[str, Any]] = []
    non_empty_lines = 0
    failed_lines = 0

    # Fast path for standard JSONL.
    for line in text.splitlines():
        if not line.strip():
            continue
        non_empty_lines += 1
        data = _parse_jsonl_line(line)
        if data:
            records.append(data)
        else:
            failed_lines += 1

    if non_empty_lines > 0 and failed_lines == 0:
        return records

    # Fallback for streams of pretty-printed JSON objects without separators.
    decoder = json.JSONDecoder()
    stream_records: List[Dict[str, Any]] = []
    idx = 0
    n = len(text)
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict):
            stream_records.append(obj)
        idx = end

    return stream_records if stream_records else records


def load_gsm8k_labels_from_parquet(parquet_path: str) -> List[str]:
    try:
        df = pd.read_parquet(parquet_path)
        labels = []
        for _, row in df.iterrows():
            ans_text = row['answer']
            if "####" in ans_text:
                label = ans_text.split("####")[-1].strip().replace(',', '')
                labels.append(label)
            else:
                labels.append(ans_text.replace(',', ''))
        return labels
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return []

def write_unified_table(results, labels, preds, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("unified table\n")
        f.write("="*80 + "\n")
        f.write(f" {'index':<5} | {'label':<30} | {'answer':<30} | {'result':<10} |\n")
        f.write("-" * 80 + "\n")
        
        for i, (is_correct, extracted_val) in enumerate(results):
            idx_str = str(i+1)
            label_str = str(labels[i]).replace('\n', ' ')
            if len(label_str) > 30: label_str = label_str[:27] + "..."
            
            ans_str = str(extracted_val).replace('\n', ' ') if extracted_val is not None else "None"
            if len(ans_str) > 30: ans_str = ans_str[:27] + "..."
            
            res_str = "right" if is_correct else "wrong"
            
            f.write(f" {idx_str:<5} | {label_str:<30} | {ans_str:<30} | {res_str:<10} |\n")

def load_math500_labels(json_path: str) -> List[str]:
    try:
        data = _read_jsonl_objects(json_path)
        labels = []
        for item in data:
            if 'output' in item:
                labels.append(item['output'])
            elif 'answer' in item:
                labels.append(item['answer'])
            elif 'solution' in item:
                labels.append(item['solution'])
        return labels
    except Exception as e:
        print(f"Error loading MATH500 json: {e}")
        return []

def load_minerva_labels(json_path: str) -> List[str]:
    try:
        data = _read_jsonl_objects(json_path)
        labels = [item.get('output', item.get('answer', item.get('solution', ''))) for item in data]
        return labels
    except Exception as e:
        print(f"Error loading Minerva json: {e}")
        return []
    
def load_aime24_labels(json_path: str) -> List[str]:
    try:
        data = _read_jsonl_objects(json_path)
        labels = [item.get('output', item.get('answer', '')) for item in data]
        return labels
    except Exception as e:
        print(f"Error loading AIME json: {e}")
        return []

def load_olympiad_labels(parquet_path: str) -> List[str]:
    try:
        df = pd.read_parquet(parquet_path)
        labels = []
        for _, row in df.iterrows():
            if 'final_answer' in row:
                val = row['final_answer']
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    labels.append(str(val[0])) 
                else:
                    labels.append(str(val))
            else:
                labels.append("")
        return labels
    except Exception as e:
        print(f"Error loading Olympiad parquet: {e}")
        return []

def load_college_math_labels(parquet_path: str) -> List[str]:
    try:
        df = pd.read_parquet(parquet_path)
        labels = df['answer'].tolist()
        return labels
    except Exception as e:
        print(f"Error loading College Math parquet: {e}")
        return []

def get_acc(
    source: Union[str, Dict[str, List]],
    dataset_type: str = "",
    predict_key: str = "predict",
    label_key: str = "label",
    verbose: bool = False,
    log_wrong_to_csv: bool = True,
    wrong_csv_path: Optional[str] = None,
    max_candidates: int = 1,
    log_dir: Optional[str] = None,
    equiv_mode: str = "legacy",
    use_gold_patch: bool = False,
) -> Tuple[float, int]:
    
    all_results = []
    preds = []
    labels = []
    questions = []
    source_path: Optional[str] = None
    
    ground_truth_labels = []
    use_ground_truth = False
    
    if isinstance(source, str):
        source_path = source
        if not dataset_type:
            dataset_type = get_dataset_type(path=source)
        print(f"use dataset_type: {dataset_type}")
        
        if "gsm8k" in dataset_type and os.path.exists(GSM8K_PARQUET_PATH):
             print(f"Loading ground truth labels from {GSM8K_PARQUET_PATH}")
             ground_truth_labels = load_gsm8k_labels_from_parquet(GSM8K_PARQUET_PATH)
        elif dataset_type == "math_500":
            path = MATH500_JSONL_PATH
            if os.path.exists(path):
                print(f"Loading ground truth labels from {path}")
                ground_truth_labels = load_math500_labels(path)
        elif dataset_type == "minerva":
            path = MINERVA_JSONL_PATH
            if os.path.exists(path):
                print(f"Loading ground truth labels from {path}")
                ground_truth_labels = load_minerva_labels(path)
        elif dataset_type in ["aime", "aime24"]:
            path = AIME_JSONL_PATH
            if os.path.exists(path):
                print(f"Loading ground truth labels from {path}")
                ground_truth_labels = load_aime24_labels(path)

        if ground_truth_labels:
            use_ground_truth = True

        with open(source, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                data = _parse_jsonl_line(line)
                if not data:
                    continue
                preds.append(data[predict_key])
                questions.append(data.get("question", data.get("problem", "")))
                
                if use_ground_truth:
                    if i < len(ground_truth_labels):
                        labels.append(ground_truth_labels[i])
                    else:
                        labels.append(data.get(label_key, ""))
                else:
                    labels.append(data.get(label_key, ""))
    
    elif isinstance(source, dict):
        assert dataset_type
        preds = source[predict_key]
        labels = source[label_key]
        questions = source.get("question", [""] * len(preds))
    else:
        raise ValueError(f"Unknown type of source: {type(source)}")

    assert preds and labels and len(preds) == len(labels)

    patched_info = []
    if use_gold_patch:
        patches = _load_gold_patches(dataset_type)
        if patches:
            labels, patched_info = _apply_gold_patches(labels, patches)
            for pid, old_l, new_l in patched_info:
                print(f"  [Gold Patch] #{pid}: '{old_l}' -> '{new_l}'")

    detailed_results = auto_verify_detailed(preds, labels, questions, dataset_type, verbose=verbose, max_candidates=max_candidates, equiv_mode=equiv_mode)
    
    correct_count = sum(1 for r in detailed_results if r.is_correct)
    acc = correct_count / len(detailed_results) if detailed_results else 0.0

    if source_path and log_wrong_to_csv:
        if log_dir is None:
            if equiv_mode == "legacy":
                log_dir = os.path.join(os.path.dirname(source_path), "logs_legacy")
            else:
                log_dir = os.path.join(os.path.dirname(source_path), f"logs_max_{max_candidates}")
        os.makedirs(log_dir, exist_ok=True)
        stem = os.path.basename(source_path).rsplit(".", 1)[0]
        
        all_csv_path = os.path.join(log_dir, f"all_results_{stem}.csv")
        write_detailed_csv(detailed_results, all_csv_path, filter_type="all")
        
        wrong_csv_path_real = wrong_csv_path or os.path.join(log_dir, f"wrong_cases_{stem}.csv")
        write_detailed_csv(detailed_results, wrong_csv_path_real, filter_type="wrong")

        right_csv_path = os.path.join(log_dir, f"right_cases_{stem}.csv")
        write_detailed_csv(detailed_results, right_csv_path, filter_type="correct")
        
        log_path = os.path.join(log_dir, "detailed_eval.log")
        write_detailed_log(detailed_results, log_path, source_path, dataset_type)
        
        log_path_old = os.path.join(log_dir, "new.log")
        results_for_old_log = [(r.is_correct, r.extracted_answer) for r in detailed_results]
        write_unified_table(results_for_old_log, labels, preds, log_path_old)
        print(f"unified table log: {log_path_old}")

        print(f"All Results CSV: {all_csv_path}")
        print(f"Wrong Cases CSV: {wrong_csv_path_real}")
        print(f"Right Cases CSV: {right_csv_path}")
        print(f"Detailed Log: {log_path}")

    return acc, len(preds)

def get_acc_per_interval(path: str, dataset_type: str = "", interval: int = 8, predict_key: str = "predict", label_key: str = "label", log_to_file: bool = True, max_candidates: int = 1) -> List[float]:
    preds, labels = [], []

    if not dataset_type:
        dataset_type = get_dataset_type(path=path)

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            data = _parse_jsonl_line(line)
            if not data:
                continue
            preds.append(data[predict_key])
            labels.append(data[label_key])
    
    num_iterations = (len(lines) // interval) if len(lines) % interval == 0 else (len(lines) // interval + 1)

    if log_to_file:
        file = open(os.path.dirname(path) + "/acc_recorder", "w", encoding="utf-8")

    results = []
    for i in range(num_iterations):
        batch_preds = preds[:(i+1)*interval]
        batch_labels = labels[:(i+1)*interval]
        batch_inputs = {
            predict_key: batch_preds,
            label_key: batch_labels
        }
        acc, num_samples = get_acc(batch_inputs, dataset_type, predict_key, label_key, max_candidates=max_candidates)
        results.append(acc)

        if log_to_file:
            file.write(f"Total {num_samples} samples, accuracy: {acc}\n")
            file.flush()

    return results

def get_dataset_type(path: str):
    dataset_type = ""
    if "gsm" in path:
        dataset_type = "gsm8k1"
    elif "math_500" in path or "math500" in path:
        dataset_type = "math_500"
    elif "aime" in path:
        dataset_type = "aime"
    elif "minerva" in path:
        dataset_type = "minerva"
    else:
        print(f"Warning: Unknown dataset type for path: {path}, defaulting to gsm8k1")
        dataset_type = "gsm8k1"
    
    return dataset_type

if __name__ == "__main__":
    paths = [
        # please paste your absolute path to result.jsonl here, e.g. "/home/user/xxx/result.jsonl"
        # the path name is used to determine the dataset type, so please keep the dataset name in the path, e.g. 'gsm8k_test_formatted', 'math_500', 'aime', 'minerva'
        "/disk/zhengkaiwen/code/tfacl/results/gsm_sota/result.jsonl"
    ]

    for path in paths:
        dataset_type = get_dataset_type(path)
        acc, num_samples = get_acc(
            source=path,
            dataset_type=dataset_type,
            equiv_mode="legacy",
            use_gold_patch=True,
            verbose=False,
        )
        print(f"Path: {path} | Acc: {acc:.4f} | Samples: {num_samples}")
