import json
import re
from llm_utils.gpt_class import GPTLLM


class LLMEvaluator(GPTLLM):
    def __init__(self, api_key=None, model=None, temperature=0.2):
        super().__init__(api_key=api_key, model=model, temperature=temperature)

    def alignment_check(self, problem_description, code):
        prompt = f"""You are an expert in optimization problem analysis. Your task is to evaluate whether the implementation code correctly models the problem description:

[Problem Description]: {problem_description}
[Implementation Code]: {code}

1. Correctness: Does the code correctly define the sets, parameters, and objective function as stated in the problem?
2. Completeness: Are all necessary constraints from the problem description fully implemented?
3. Solvability: Can the code reliably solve the problem type and address its specific modeling challenges?

Respond in exactly this format:
ALIGNED: [True/False]
EXPLANATION: [One sentence stating the main alignment issue OR confirming the code correctly models the problem]
Focus on whether the code matches the problem specification, not code quality or efficiency.
"""
        return self._call_api(prompt)

    def feasibility_check(self, problem_description, opt_var):
        prompt = f"""You are a solution feasibility validator. Given a problem description and proposed solution:
        [Problem Description]: {problem_description}
        [Solution]: {opt_var}

Identify the 2-3 key constraints that define solution validity from the problem description
Check if the proposed solution satisfies these constraints
Respond in exactly this format:

FEASIBLE: [True/False]
EXPLANATION: [One sentence stating which constraint is violated OR confirming all key constraints are satisfied]
Focus only on feasibility (constraint satisfaction), not solution optimality.
"""
        return self._call_api(prompt)

    def solution_feasibility_check(
        self,
        problem_description,
        forward_instruction,
        assumption,
        math_formulation,
        opt_variables,
    ):
        prompt = f"""You are an expert in optimization problem analysis. Your task is to determine the feasibility of a proposed solution for a given optimization problem.

        **Given Inputs:**
        - An optimization problem description in natural language
        - **Key modeling instructions** that clarify or extend the problem description
        - The **mathematical formulation** of the problem, including variable definitions and constraints
        - A **proposed solution**, presented as a dictionary of decision variable values

        [Problem Description and Assumptions]
        {problem_description}
        {assumption}

        [Key Modeling Instructions]
        {forward_instruction}

        [Mathematical Formulation]
        {math_formulation}

        [Proposed Solution]
        {opt_variables}

        **Task:**
        First, verify that all parameters are correctly introduced in the mathematical formulation, ensuring there are no new, missing, or incorrect values introduced (except those explicitly indicated in the problem statement).
        Then analyze the proposed solution to determine if it is **feasible**. A solution is feasible if it satisfies **all structural constraints** from the mathematical formulation and key modeling instructions. 

        **IMPORTANT**: **Do NOT check variable domain constraints** (such as binary, integer, or non-negativity restrictions) as numerical precision issues can cause false negatives. Focus only on the problem's structural constraints (equalities, inequalities, and logical relationships).

        **Evaluation Guidelines:**

        **Numerical Tolerance:**
        - Use tolerance of 1e-6 for all floating-point comparisons
        - Equality constraints: |actual - target| ≤ 1e-6
        - Inequality constraints: allow 1e-6 slack beyond bounds
        - Treat -0.0 as equivalent to 0.0

        **Evaluation Process:**
        1. Extract ALL structural constraints from the mathematical formulation (excluding variable domain constraints)
        2. **Show your detailed calculations** for each constraint evaluation
        3. **Double-check arithmetic** before determining constraint satisfaction
        4. Check satisfaction using numerical tolerance
        5. Account for modeling instructions that affect constraint interpretation

        **Output Format:**
        First, show your detailed work and calculations for evaluating each constraint.
        Then, conclude with the keyword **Summary** followed by the JSON object:

        **Summary**
        {{
        "feasible": <true or false>,
        "violated_constraints": ["Detailed descriptions of violated/non-evaluable structural constraints (empty if feasible)"],
        "checked_constraints": ["Descriptions of all structural constraints evaluated"]
        }}
        """

        default_result = {
            "feasible": False,
            "violated_constraints": [],
            "checked_constraints": [],
        }

        content, prob = self._call_api_conf(prompt, default_result)
        if not prob:
            content, prob = self._call_api_conf(prompt, default_result)
            if not prob:
                prob = 0.0
        return content, prob

    def verify_optimality(self, execution_output, ground_truth):
        """
        Verify if the optimization solution matches the ground truth.
        Returns a (is_correct, explanation) tuple.
        """
        prompt = f"""
        I need you to verify if an optimization model solution is correct.

        Execution output from the GurobiPy model:
        ```
        {execution_output}
        ```

        Ground truth value (expected optimal value): {ground_truth}

        Please verify if the optimization solution matches the ground truth.
        Answer with "YES" or "NO" followed by a brief explanation. 
        Do not output anything other than your verification and explanation.

        IMPORTANT: 
        1. As long as the result is within 1% of the ground truth value (i.e., |execution result - ground truth|/ ground truth < 0.01), the output should be YES and an explanation for why it is YES
        2. As long as the optimization solution is consistent with the ground truth, the output should be YES, and no additional analysis is required
        3. If the result can be rounded or truncated to a result that differs from ground by 0.001, then the output should also be YES and an explanation for why it is YES
        """
        content = self._call_api(prompt)

        if not content:
            return False, "No response from API."

        # Check for YES/NO answer
        if "YES" in content.upper():
            return True, content
        elif "NO" in content.upper():
            return False, content
        else:
            return False, "Unexpected response format."
