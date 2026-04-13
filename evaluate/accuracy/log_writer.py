import csv
import os
from datetime import datetime
from typing import List
from acc_utils import EvalResult

def _escape_for_csv(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    return text.replace("\r\n", "\n").replace("\r", "\n")

def _truncate(s: str, max_len: int) -> str:
    if s is None:
        return ""
    if len(s) <= max_len:
        return s
    return s[:max_len-3] + "..."

def _indent(text: str, spaces: int) -> str:
    if not text:
        return ""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.split("\n"))

def write_detailed_csv(
    results: List[EvalResult],
    output_path: str,
    filter_type: str = "all"
):
    fieldnames = [
        "idx", 
        "result", 
        "ground_truth", 
        "ground_truth_normalized",
        "extracted_answer", 
        "extracted_normalized",
        "match_reason", 
        "match_detail",
        "question", 
        "response"
    ]
    
    rows_to_write = []
    for res in results:
        if filter_type == "correct" and not res.is_correct:
            continue
        if filter_type == "wrong" and res.is_correct:
            continue
            
        response_text = res.response
        if response_text:
            words = response_text.split()
            if len(words) > 40:
                response_text = "... " + " ".join(words[-40:])

        row = {
            "idx": res.idx,
            "result": "✓" if res.is_correct else "✗",
            "ground_truth": _escape_for_csv(res.ground_truth),
            "ground_truth_normalized": _escape_for_csv(res.ground_truth_normalized),
            "extracted_answer": _escape_for_csv(res.extracted_answer),
            "extracted_normalized": _escape_for_csv(res.extracted_normalized),
            "match_reason": res.match_reason.value,
            "match_detail": _escape_for_csv(res.match_detail),
            "question": _escape_for_csv(res.question),
            "response": _escape_for_csv(response_text)
        }
        rows_to_write.append(row)
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
        
    print(f"CSV written: {output_path} ({len(rows_to_write)} rows)")

def write_detailed_log(
    results: List[EvalResult],
    output_path: str,
    source_path: str,
    dataset_type: str
):
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    wrong = total - correct
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    reason_stats = {}
    for r in results:
        reason = r.match_reason.value
        reason_stats[reason] = reason_stats.get(reason, 0) + 1
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("EVALUATION REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Source:      {source_path}\n")
        f.write(f"Dataset:     {dataset_type}\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total:       {total}\n")
        f.write(f"Correct:     {correct}\n")
        f.write(f"Wrong:       {wrong}\n")
        f.write(f"Accuracy:    {accuracy:.2f}%\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("MATCH REASON STATISTICS\n")
        f.write("=" * 100 + "\n")
        sorted_reasons = sorted(reason_stats.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons:
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"  {reason:<25} {count:>5}  ({pct:>5.1f}%)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("=" * 100 + "\n")
        f.write(f"WRONG CASES SUMMARY ({wrong} total)\n")
        f.write("=" * 100 + "\n")
        wrong_cases = [r for r in results if not r.is_correct]
        for r in wrong_cases:
            gt_short = _truncate(str(r.ground_truth), 30).replace("\n", "\\n")
            ext_short = _truncate(str(r.extracted_answer), 30).replace("\n", "\\n")
            f.write(f"idx={r.idx:<4}: GT={gt_short:<32}, EXT={ext_short:<32}, reason={r.match_reason.value}\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("DETAILED RESULTS (sorted by idx)\n")
        f.write("=" * 100 + "\n")
        
        for r in results:
            status = "✓ CORRECT" if r.is_correct else "✗ WRONG"
            f.write(f"\n[{r.idx:03d}] {status}\n")
            f.write(f"  Ground Truth:    {_indent(str(r.ground_truth), 19).strip()}\n")
            f.write(f"  Extracted:       {str(r.extracted_answer)}\n")
            f.write(f"  Match Reason:    {r.match_reason.value} ({r.match_detail})\n")
            if not r.is_correct:
                resp_preview = _truncate(r.response, 100).replace("\n", "\\n")
                f.write(f"  Response Preview: {resp_preview}\n")

    print(f"Log written: {output_path}")
