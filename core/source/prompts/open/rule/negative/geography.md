You are a failure analyst for evaluation. Below are {n} low-scoring question-answer pairs that deviate significantly from the reference answers—not due to factual errors, but because of *stylistic overreach*.

Focus your analysis on:
- **Length correlation**: Does the answer length scale with the question length? Is it deliberately short?
- **Tone**: Is the response casual, direct, or opinionated—avoiding academic or instructional phrasing?
- **Formatting**: Does it avoid markdown, bullet points, numbered steps, section headers, or bold/italic text?

Requirements:
1. The rule must guide *how to write*, not *what to say*. Prioritize constraints on tone, structure, and length.
2. If applicable, phrase the rule as: "**IF** [question property, e.g., short/informal/opinion-based], **THEN** [output constraint, e.g., answer in one plain sentence]."
3. Keep the rule concise (1–2 sentences).
4. Output ONLY the rule. No explanation, no prefix.
  
Question-and-answer pair list:  
{qa_pairs}  
  
Please extract the positive experience rule (Output ONLY the rule logic):  