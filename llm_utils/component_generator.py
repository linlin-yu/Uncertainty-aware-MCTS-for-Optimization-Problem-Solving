from typing import Dict, List, Optional, Tuple, Any
from llm_utils.gpt_class import GPTLLM
import random


class EfficientLLMComponentGenerator(GPTLLM):
    """
    Generate optimization formulation components on demand using LLM API calls.

    This class progressively builds mathematical optimization problems by generating
    sets, parameters, variables, objectives, and constraints using Large Language
    Model API calls with probability tracking.
    """

    # Class constants
    LEVEL_ROOT, LEVEL_TYPE, LEVEL_INSTRUCTION = 0, 1, 2
    LEVEL_SETPARAM, LEVEL_MODEL, LEVEL_CODE = 3, 4, 5
    SOLVER_LIST = ["gurobipy", "pyomo_cbc", "pyomo_scip"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.8,
    ):
        """Initialize the component generator."""
        super().__init__(api_key=api_key, model=model, temperature=temperature)

    def check_entailment(self, problem_description, formulation_A, formulation_B):
        """Check semantic entailment between two texts."""
        prompt = f"""
You are an expert in optimization modeling. You are given:

[Problem Description]: {problem_description}
[Formulation A]: {formulation_A}
[Formulation B]: {formulation_B}

Determine whether **Formulation A and Formulation B are semantically equivalent representations** 
of the same optimization problem, based on the problem description.

Use **logical entailment** as your criterion:
- A formulation is **entailed** by the problem description if it correctly represents the modeling elements implied by it.
- Two formulations are **semantically equivalent** if they capture the same sets, parameters, decision variables, objective, and constraints, even if notation differs.

Respond with exactly one word: `"yes"` if semantically equivalent and mutually entailed, `"no"` otherwise.
        """
        return 1 if "yes" in self._call_api(prompt).lower() else 0

    def generate_next_component(
        self, trajectory, level, previous_generation=None, error_message=None, solver_type=None
    ):
        """Generate the next component based on the current level in the trajectory."""
        generators = {
            self.LEVEL_TYPE: lambda: self._generate_type(
                trajectory["problem_description"]
            ),
            self.LEVEL_INSTRUCTION: lambda: self._generate_instruction(trajectory["type"]),
            self.LEVEL_SETPARAM: lambda: self._generate_setsparams(
                trajectory["problem_description"],
                trajectory["type"],
                previous_generation,
                error_message,
            ),
            self.LEVEL_MODEL: lambda: self._generate_model(
                trajectory["problem_description"],
                trajectory["instruction"],
                trajectory["setparams"],
                previous_generation,
                error_message,
            ),
            self.LEVEL_CODE: lambda: self._generate_code(
                trajectory["setparams"],
                trajectory["formulation"],
                solver_type,
                previous_generation,
                error_message,
            ),
        }

        if level not in generators:
            raise NotImplementedError(f"Level {level} is not implemented")

        return generators[level]()

    def _generate_type(
        self, problem_description: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        prompt = f"""You are given a natural language description of an optimization problem.

[Problem Description]: {problem_description}

Your task: Identify the problem type (e.g., traveling salesman, production planning, network flow, assignment).
Return the problem type with one sentence introduction."""
        return self._call_api_with_probability(prompt)

    def _generate_instruction(
        self, problem_type: str
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        prompt = f"""You are an optimization expert. Analyze this problem: {problem_type}

Provide a comprehensive modeling analysis:
1. Highlight the most important modeling challenge for this problem type.
2. Summarize the core modeling logic including decision variables, objective function, and key constraints.

Example Format:
**Problem**: Traveling Salesman Problem
**Primary Challenge**: Subtour elimination - preventing disconnected cycles
**Core Modeling Logic**:
1. **Decision Variables**: 
- Define binary decision variables x[i, j] indicating whether the path goes directly from city i to city j.  
- Define auxiliary integer variables u[i] for subtour elimination (Miller–Tucker–Zemlin formulation).
2. **Constraints**: 
- Each city must be visited exactly once (in-degree = 1, out-degree = 1).  
- Subtour elimination: u[i] - u[j] + n * x[i, j] <= n - 1 for all i != j, i != start, j != start.  
- Fix starting point: choose one city (e.g., 'A') and set u['A'] = 0.  
- Bounds: u[i] in [1, n-1] for all i != start.  
3. **Objective**: Minimize total travel cost"""
        return self._call_api_with_probability(prompt)

    def _generate_setsparams(
        self,
        problem_text: str,
        problem_type: str,
        previous_generation: str,
        error_message: str,
    ):
        prompt = f"""You are an optimization expert.

[Problem Text]: {problem_text}
{problem_type}

Extract and define the **Sets** and **Parameters** for this mathematical formulation.
"""
        if error_message:
            prompt += f"Previous generation: {previous_generation}\nError: {error_message}\nUse for correction.\n"

        prompt += """
## Part 1: Sets
Return as Python list of dictionaries with 'name' and 'description' keys.

## Part 2: Parameters  
Return as Python list of dictionaries:
- Indexed parameters: 'name', 'description', 'index_set', 'values' keys
- Scalar parameters: 'name', 'description', 'value' keys

IMPORTANT: No markdown formatting or code blocks. Return only the required structure."""
        return self._call_api_with_probability(prompt)

    def _generate_model(
        self,
        problem_text: str,
        problem_instruction: str,
        sets_parameters: str,
        previous_generation: str,
        error_message: str,
    ):
        prompt = f"""Complete the mathematical formulation by defining decision variables, objective function, and constraints.

[Problem Description]: {problem_text}
[Modeling Instructions]: {problem_instruction}
[Given Sets and Parameters]: {sets_parameters}
"""
        if error_message:
            prompt += f"Previous generation: {previous_generation}\nError: {error_message}\nUse for correction.\n"

        prompt += """
## Part 3: Decision Variables
Python list of dictionaries with 'name', 'description', 'index_set', 'domain', 'interpretation' keys. Note use the sets defined in the given Sets and Parameters section.

## Part 4: Objective Function
Python dictionary with 'type' (minimize/maximize), 'expression', 'description' keys.

## Part 5: Constraints
Python list of dictionaries with 'name', 'mathematical_form', 'description', 'constraint_type' keys.

IMPORTANT: No markdown formatting, explanation or code blocks."""
        return self._call_api_with_probability(prompt)

    def _generate_code(
        self,
        sets_parameters: str,
        formulation: str,
        solver: str,
        previous_generation: Optional[str] = None,
        error_message: Optional[str] = None,
    ):
        """Generate executable optimization code from mathematical formulation."""
        generators = {
            "gurobipy": self._generate_gurobipy_code,
            "pyomo_cbc": self._generate_pyomo_cbc_code,
            "pyomo_scip": self._generate_pyomo_scip_code,
        }

        if solver not in generators:
            raise ValueError(f"Unsupported solver type: {solver}")

        code, sentence_entropy, sentence_logprob = generators[solver](
            sets_parameters, formulation, previous_generation, error_message
        )

        # Clean markdown formatting
        if code.startswith("```") and "```" in code[3:]:
            start = code.find("\n", code.find("```"))
            end = code.find("```", start)
            if end > start:
                code = code[start + 1 : end].strip()

        # Clean quotes and escape sequences
        if code.startswith('"') and code.endswith('"'):
            code = code[1:-1]
        code = code.encode().decode("unicode_escape")

        return code, sentence_entropy, sentence_logprob

    def _generate_solver_code(
        self,
        solver_type: str,
        sets_parameters: str,
        formulation: str,
        previous_code: str = None,
        error_message: Optional[str] = None,
    ):
        """Common code generation logic for all solvers."""
        solver_configs = {
            "gurobipy": {
                "import": "from gurobipy import *",
                "model_creation": "Model()",
                "solver": "gruobipy",
                "optimize": ".optimize()",
            },
            "pyomo_cbc": {
                "import": "from pyomo.environ import *",
                "model_creation": "concrete model",
                "solver": "'cbc' for linear problems and 'ipopt' for nonlinear problems",
                "optimize": ".optimize()",
            },
            "pyomo_scip": {
                "import": "from pyomo.environ import *",
                "model_creation": "concrete model",
                "solver": "'scip'",
                "optimize": ".optimize()",
            },
        }

        config = solver_configs[solver_type]
        prompt = f"""Generate complete, executable {solver_type} code for this optimization problem.

[Problem Formulation]: {sets_parameters}, {formulation}
"""
        if error_message:
            prompt += f"Previous code: {previous_code}\nError: {error_message}\nFix the specific error.\n"

        prompt += f"""
Requirements:
1. Use `{config["import"]}` as import
2. Create a {config["model_creation"]}
3. Define all sets, parameters, variables, objective, constraints from formulation
4. **CRITICAL - Avoid Index Errors**: Only create variables for valid index combinations
- Only create variables for valid index combinations that exist in your defined sets/parameters
- When using summation in constraints or objective, always check if the index combination exists before referencing it
- Use conditional logic like `if (i,j) in valid_pairs` or `if parameter[i][j] > 0` before accessing variables
- For network problems, only reference edges that actually exist in the network
- Do not use chained inequalities like `a <= x <= b`; instead, add separate constraints for lower and upper bounds or use `addRange`.
5. Use {config["solver"]} solver
6. Call {config["optimize"]} to solve
7. Output logic:
   - If optimal: Print "Optimal solution found", "Objective Value: <value>", "BEGIN_VARIABLES", 
     variable values, "END_VARIABLES"
   - If infeasible/unbounded: Print "Model is infeasible or unbounded"

Return only raw Python code, no markdown or explanations."""
        return self._call_api_with_probability(prompt)

    def _generate_gurobipy_code(
        self,
        sets_parameters: str,
        formulation: str,
        previous_code: str = None,
        error_message: Optional[str] = None,
    ):
        """Generate GurobiPy implementation code."""
        return self._generate_solver_code(
            "gurobipy", sets_parameters, formulation, previous_code, error_message
        )

    def _generate_pyomo_cbc_code(
        self,
        sets_parameters: str,
        formulation: str,
        previous_code: str = None,
        error_message: Optional[str] = None,
    ):
        """Generate Pyomo implementation code with CBC solver."""
        return self._generate_solver_code(
            "pyomo_cbc", sets_parameters, formulation, previous_code, error_message
        )

    def _generate_pyomo_scip_code(
        self,
        sets_parameters: str,
        formulation: str,
        previous_code: str = None,
        error_message: Optional[str] = None,
    ):
        """Generate Pyomo implementation code with SCIP solver."""
        return self._generate_solver_code(
            "pyomo_scip", sets_parameters, formulation, previous_code, error_message
        )
