import re
from llm_utils.gpt_class import GPTLLM


class DirectGPT(GPTLLM):
    def __init__(self, api_key=None, model=None, temperature=0.2):
        """
        Initialize the optimization-focused LLM.

        Args:
            api_key (str): OpenAI API key
            model (str): Model name to use
            temperature (float): Sampling temperature (lower for more consistent code)
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature)

    def _build_prompt(self, problem_text):
        """
        Private method to construct an optimization problem prompt.

        Args:
            problem_text (str): The optimization problem description

        Returns:
            str: The formatted prompt for optimization problem solving
        """
        prompt = f"""Suppose you are an optimization problem expert, please help me solve the following problems, specifically, first generate the mathematical model, then generate the gurobipy code.

                Problem:
                {problem_text}

                Please structure your response as follows:
                1. Mathematical Model: Clearly define the decision variables, objective function, and constraints
                2. Gurobi Python Code: Provide complete, runnable gurobipy code enclosed in ```python code blocks

                Make sure the code is complete and can be executed directly."""

        return prompt

    def _generate_response(self, problem_text):
        """
        Generate a response for an optimization problem.

        Args:
            problem_text (str): The optimization problem description
            max_retries (int): Maximum number of retry attempts on API failure

        Returns:
            str: The generated response, or None if all attempts failed
        """

        # Build the prompt using the private method
        formatted_prompt = self._build_prompt(problem_text)

        response = self._call_api(formatted_prompt)
        if response:
            return response
        else:
            return None

    def _extract_code(self, response):
        """
        Extract Gurobi Python code from the LLM response.

        Args:
            response (str): The complete response from the LLM

        Returns:
            str: The extracted Python code, or None if no code found
        """

        # Look for code blocks marked with ```python
        python_code_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(python_code_pattern, response, re.DOTALL)

        if matches:
            # Return the first (or concatenate all) code blocks
            return matches[0].strip()

        # Fallback: look for code blocks without language specification
        code_pattern = r"```\s*(.*?)\s*```"
        matches = re.findall(code_pattern, response, re.DOTALL)

        if matches:
            # Filter for blocks that look like Python code (contain gurobi imports)
            for match in matches:
                if any(
                    keyword in match.lower()
                    for keyword in [
                        "import gurobipy",
                        "from gurobipy",
                        "gurobi",
                        "model.optimize",
                    ]
                ):
                    return match.strip()

        return None

    def forward(self, problem_text):
        """
        Solve an optimization problem and return both the full response and extracted code.

        Args:
            problem_text (str): The optimization problem description
            max_retries (int): Maximum number of retry attempts on API failure

        Returns:
            dict: Dictionary containing 'response' and 'code' keys
        """
        results = {"formulation": "", "code": "", "success": False}
        response = self._generate_response(problem_text)
        if response is not None:
            code = self._extract_code(response)
            if code is not None:
                results["success"] = True
        else:
            code = None
        results["formulation"] = response
        results["code"] = code
        return results
