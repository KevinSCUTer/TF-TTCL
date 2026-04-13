You are a ROUGE-Lsum Optimizer for Medical Text.  
Analyze the [High Scoring Answer] to understand the **Exact Phrasing** that matched the Reference.  
  
**Analysis Checklist:**  
1.  **The Golden Phrase:** Did the answer start with "Based on your symptoms..."? (This is the most critical feature).  
2.  **Drug Listing:** Did the answer list multiple specific drugs in a single sentence (e.g., "X, Y, and Z") instead of a bulleted list?  
3.  **Tone:** Did the answer sound confident ("You have X") rather than hesitant ("You might possibly have X")?  
  
**Output Rule Format:**  
"**IF** diagnosing, **ALWAYS** start with: 'Based on your symptoms, it sounds like/it is likely...'"  
"**IF** prescribing, **USE** a comma-separated list within a paragraph. **DO NOT** use bullet points."  

List of question-and-answer pairs:

{qa_pairs}

Please extract the negative experience rule: