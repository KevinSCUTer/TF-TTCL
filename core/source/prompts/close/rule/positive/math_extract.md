<task>
Analyze the provided {n} examples of INCORRECT mathematical derivations.
Identify the **Root Cause** of the failure. Specifically, look for:
1. **Silent Assumptions**: Assuming continuity, invertibility, or integers without proof.
2. **Boundary Violations**: Ignoring domain restrictions ($x \neq 0$, $\ln(x)$ where $x>0$).
3. **Logical Gaps**: Confusing sufficiency with necessity.

Synthesize a "Negative Rule" that explicitly warns against the **Trigger Condition** that leads to the error.
</task>

<examples>
{qa_pairs}
</examples>

<requirements>
1. Format: "When [Specific Context/Condition], do NOT assume/forget [Trap/Constraint]."
2. Strict limit: Under 45 words.
3. Be abstract: Use mathematical terms (e.g., "discriminant", "convexity") rather than specific numbers.
4. If the error is a specific calculation slip, generalize it to the operation type (e.g., "matrix multiplication order").
</requirements>

Negative Rule: