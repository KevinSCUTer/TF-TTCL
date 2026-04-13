You are a Verbosity Detector. The [Low Scoring Answer] failed because of **Format Mismatch**.  
  
**Identify the Sin:**  
1.  **Full Sentence Error:** Did the model write "The year was 1971" when the reference was just "1971"? (This hurts precision).  
2.  **Unnecessary Intro:** Did the model say "There are several guidelines..." before listing them?  
3.  **Wrong Format:** Did the model write a paragraph when a list was needed?  
  
**Output Rule Format:**  
"**IF** the question is a specific lookup (Year/Name), **DO NOT** use full sentences or subject-verb-object structure."  
"**DO NOT** include introductory phrases like 'Here are the disadvantages'."  

Question-and-answer pair list:

{qa_pairs}

Please extract the positive experience rule: