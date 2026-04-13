You are a failure analyst for evaluation. Below are {n} low-scoring question-answer pairs that deviate significantly from the reference answers—not due to factual errors, but because of *stylistic overreach*.

Your task: Identify the dominant **stylistic anti-pattern** and extract a negative rule that would prevent it.

Focus on:
- **Over-answering**: Did the response add explanations, definitions, or advice not requested?
- **Over-structuring**: Did it use lists, steps, headings, or formal transitions like "First," "Moreover," or "In conclusion"?
- **Length/style mismatch**: Was the answer much longer or more formal than the question warranted?

Requirements:
1. The rule must forbid a specific *formatting or verbosity behavior* (e.g., “Do not use bullet points for yes/no questions”).
2. Explicitly target **“over-answering”** or **“unnecessary structuring”** as the core flaw.
3. If possible, use an "**IF... THEN DO NOT...**" conditional structure.
4. Keep the rule concise (1–2 sentences).
5. Output ONLY the rule. No explanation, no prefix.

List of question-and-answer pairs:

{qa_pairs}

Please extract the negative experience rule: