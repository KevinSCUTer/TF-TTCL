<task>  
Analyze the Question, the High-Quality Answer, and the Low-Quality Answer.  
Identify the specific flaw (e.g., verbosity, hallucination, wrong tone, formatting error) in the Low-Quality Answer.  
Extract a concise "Negative Rule" to prevent this mistake in the future.  
</task>  
  
<question>  
{question}  
</question>  
  
<correct_answer>  
{correct_answer}  
</correct_answer>  
  
<incorrect_answer>  
{incorrect_answer}  
</incorrect_answer>  
  
<requirements>  
1. Start with "Avoid" or "Do not".  
2. Keep it strictly under 32 words. 
3. Focus on target specific open-domain errors. e.g., "Avoid repeating the user's question" or "Avoid verbose introductions".  
4. The rule must help constraint word count and increase precision.  
</requirements>  
  
Negative Rule: