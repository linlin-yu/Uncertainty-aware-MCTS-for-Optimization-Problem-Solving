from llm_utils.gpt_class import GPTLLM


class SequentialGPT(GPTLLM):
    def __init__(self, api_key=None, model=None, temperature=0.2):
        """
        Initialize the optimization-focused LLM.

        Args:
            api_key (str): OpenAI API key
            model (str): Model name to use
            temperature (float): Sampling temperature (lower for more consistent code)
        """
        super().__init__(api_key=api_key, model=model, temperature=temperature)

    def _generate_sets(self, problem_text):
        prompt = f"""
        Suppose you are an optimization problem expert. You are given an optimization problem in natural language:

        [Problem Text]: {problem_text}

        Step 1: Identify and define all sets involved in the problem. A set usually corresponds to a group or category of entities, such as time periods, products, facilities, workers, locations, etc. Return your response as a Python list of dictionaries.
        Each dictionary should have 'name', 'dimen' and 'elements' keys. Don't include any explanations.

        IMPORTANT: Do not use markdown formatting or code blocks. Return only the raw Python list.
        If there are too many elements, such as Worker1, Worker2, ..., Worker1000
        Please use '...' The omitted part should be defined as: ['Worker1', 'Worker2', '...', 'Worker1000'],

        For example:
        [
            {{
                "name": "Products", 
                "dimen": dimensions of this set,
                "elements": ["Product1", "Product2", "Product3"]
            }}
        ]

        IMPORTANT Special Note:
        For graph-based problems, it is important to model not only the nodes but also the edges as a set. 
        For example,  for a graph-based problem with 8 nodes, the return should be:
        [
            {{
                "name": "Nodes", 
                "dimen": 1,
                "elements": [1, 2, 3, 4, 5, 6, 7, 8]
            }},
            {{
                "name": "Edges", 
                "dimen": 2,
                "elements": [
                    (1, 2), (1, 3),
                    (2, 1), (2, 4), (2, 3),
                    (3, 1), (3, 2), (3, 5), (3, 6),
                    (4, 2), (4, 6),
                    (5, 3), (5, 8),
                    (6, 3), (6, 4), (6, 7),
                    (7, 6), (7, 8),
                    (8, 5), (8, 7)
                ]
            }}
        ]
        """
        response = self._call_api(prompt)
        return self._parse_llm_response(response)

    def _generate_parameters(self, problem_text, sets):
        prompt = f"""
        Suppose you are an optimization expert. You are given an optimization problem in natural language:

        [Problem Text]: {problem_text}

        The following sets have been identified:
        [Previously generated sets]: {sets}

        Step 2: Identify all known numerical inputs in the problem and define them as parameters. Return your response as a Python list of dictionaries.
        For indexed parameters, include 'name', 'index_set', and 'values' keys.
        For scalar parameters, include 'name' and 'value' keys.
        Don't include any explanations.

        IMPORTANT: Do not use markdown formatting or code blocks. Return only the raw Python list.

        For example:
        [
            {{
                "name": "Cost",
                "index_set": "Products",
                "values": {{"Product1": 10, "Product2": 15, "Product3": 20}}
            }},
            {{
                "name": "Capacity",
                "value": 100
            }}
        ]

        Extra requirements, please follow them strictly:
        1.  Do not use tuples or lists as dictionary keys. If a key is a tuple or list, convert it into a string instead. For example, 
            {{('SparePart1', 'Machine1'): 2}}
            should be rewritten as:
            {{'SparePart1, Machine1': 2}}
        """
        response = self._call_api(prompt)
        return self._parse_llm_response(response)

    def _generate_variables(self, problem_text, sets, parameters):
        prompt = f"""
        You are given an optimization problem in natural language:

        [Problem Text]: {problem_text}

        Previously identified sets:
        [Previously generated sets]: {sets}

        Previously identified parameters:
        [Previously generated parameters]: {parameters}

        Step 3: Define all decision variables for this problem. Return your response as a Python list of dictionaries.
        Each dictionary should have 'name', 'domain', and optionally 'index_set' and 'description' keys.
        The domain should be one of: 'Binary', 'Integer', 'NonNegativeIntegers', 'NonNegativeReals', or 'Reals'.
        Don't include any explanations.

        IMPORTANT: Do not use markdown formatting or code blocks. Return only the raw Python list.

        For example:
        [
            {{
                "name": "x",
                "domain": "Binary",
                "index_set": "Products",
                "description": "Whether to select each product"
            }}
        ]
        """
        response = self._call_api(prompt)
        return self._parse_llm_response(response)

    def _generate_objectives(self, problem_text, sets, parameters, variables):
        prompt = f"""
        You are given an optimization problem in natural language:

        [Problem Text]: {problem_text}

        Previously identified sets:
        [Previously generated sets]: {sets}

        Previously identified parameters:
        [Previously generated parameters]: {parameters}

        Previously defined decision variables:
        [Previously generated decision variables]: {variables}

        Step 4: Formulate the objective function. Return your response as a Python list with a single dictionary.
        The dictionary should have 'name', 'sense', and 'expression' keys. The sense should be either 'maximize' or 'minimize'.
        The expression should be a valid GurobiPy expression as a string, using 'model.' to reference sets, parameters, and variables.
        Don't include any explanations.

        After defining the objective, please re-substitute it back into the problem statement to check for any inconsistencies or misunderstandings. 
        For example, if the question asks for the number of products, but the objective calculates the number of orders instead, this would be incorrect.

        IMPORTANT: Do not use markdown formatting or code blocks. Return only the raw Python list.

        For example:
        [
            {{
                "name": "MaximizeProfit",
                "sense": "maximize",
                "expression": "sum(model.Revenue[p] * model.x[p] - model.Cost[p] * model.x[p] for p in model.Products)"
            }}
        ]
        """
        response = self._call_api(prompt)
        objectives = self._parse_llm_response(response)
        # Return only the first objective if multiple are generated
        return objectives[:1] if objectives else []

    def _generate_constraints(self, problem_text, sets, parameters, variables):
        prompt = f"""
        You are given an optimization problem in natural language:

        [Problem Text]: {problem_text}

        Previously identified sets:
        [Previously generated sets]:{sets}

        Previously identified parameters:
        [Previously generated parameters]: {parameters}

        Previously defined decision variables: 
        [Previously generated decision variables]: {variables}

        Step 5: Write all constraints required by the problem. Return your response as a Python list of dictionaries.
        Each dictionary should have 'name', 'expression', and optionally 'description' keys.
        The expression should be a valid Gurobipy expression as a string, using 'model.' to reference sets, parameters, and variables.
        Don't include any explanations.

        IMPORTANT: Do not use markdown formatting or code blocks. Return only the raw Python list.

        For example:
        [
            {{
                "name": "CapacityConstraint",
                "expression": "sum(model.Weight[p] * model.x[p] for p in model.Products) <= model.Capacity",
                "description": "Total weight must not exceed capacity"
            }}
        ]

        """
        response = self._call_api(prompt)
        return self._parse_llm_response(response)

    def _build_formulation_str(
        self,
        sets: str,
        parameters: str,
        decision_variables: str,
        objective: str,
        constraints: str,
    ) -> str:
        # Use plain text structure with clear section titles (no LaTeX)
        formulation = f"""Sets:
                    {sets.strip()}

                    Parameters:
                    {parameters.strip()}

                    Decision Variables:
                    {decision_variables.strip()}

                    Objective:
                    {objective.strip()}

                    Constraints:
                    {constraints.strip()}

                    """.strip()

        return formulation

    def _generate_code(self, formulation):
        prompt = f"""
            I need you to generate Gurobipy code for the following mathematical formulation.

            Formulation:
            {formulation}

            Please generate complete, executable Gurobipy code that implements this formulation. The code should:

            1. Use "from gurobipy import *" as the import command.
            2. Create a model using `Model()`.
            3. Define sets (as lists or dictionaries), parameters, variables, objective, and constraints as specified in the formulation.
            4. Solve the model using `.optimize()` and check the solver status.
            5. After solving, include the following output logic:
                - If the termination condition is optimal, print exactly:
                    "Optimal solution found"
                    "Objective Value: <value>"
                    "BEGIN_VARIABLES"
                    Then for each variable:
                    "Variable <var.name>: <value>"
                    Finally print:
                    "END_VARIABLES"
                - If the model is infeasible or unbounded, print exactly:
                    "Model is infeasible or unbounded"
                - For any other solver status, print:
                    "Solver status unclear"

            Additional instructions:
            - Use `.X` only after checking the model status confirms the solution is optimal (`model.Status == GRB.OPTIMAL`).
            - Use `model.addConstr()` or `model.addConstrs()` for constraints. Do not use Python conditionals on decision variables before solving.
            - If the model is infeasible or unbounded, print a message like "Model is infeasible or unbounded." and do not attempt to print `.X`.
            - Return only clean Python code with no markdown, explanation, or formatting characters.
            """
        response = self._call_api(prompt)
        # do not evaluate it as python code for now
        return response

    def forward(self, problem_text):
        """
        Generate a complete mathematical optimization formulation and code from a natural language problem description.

        Args:
            problem_text (str): Natural language description of the optimization problem

        Returns:
            dict: Dictionary containing:
                - 'sets': List of identified sets
                - 'parameters': List of identified parameters
                - 'variables': List of decision variables
                - 'objective': List containing the objective function
                - 'constraints': List of constraints
                - 'formulation': Complete formulation as string
                - 'code': Generated Gurobipy code
                - 'success': Boolean indicating if all steps completed successfully
                - 'errors': List of any errors encountered
        """
        result = {
            "sets": [],
            "parameters": [],
            "variables": [],
            "objective": [],
            "constraints": [],
            "formulation": "",
            "code": "",
            "success": False,
            "errors": [],
        }

        try:
            print("Step 1: Generating sets...")
            sets = self._generate_sets(problem_text)
            if not sets:
                result["errors"].append("Failed to generate sets")
                return result
            result["sets"] = sets
            print(f"Generated {len(sets)} sets")

            print("Step 2: Generating parameters...")
            parameters = self._generate_parameters(problem_text, sets)
            if not isinstance(parameters, list):
                result["errors"].append("Failed to generate parameters")
                return result
            result["parameters"] = parameters
            print(f"Generated {len(parameters)} parameters")

            print("Step 3: Generating decision variables...")
            variables = self._generate_variables(problem_text, sets, parameters)
            if not variables:
                result["errors"].append("Failed to generate variables")
                return result
            result["variables"] = variables
            print(f"Generated {len(variables)} variables")

            print("Step 4: Generating objective function...")
            objective = self._generate_objectives(
                problem_text, sets, parameters, variables
            )
            if not objective:
                result["errors"].append("Failed to generate objective")
                return result
            result["objective"] = objective
            print("Generated objective function")

            print("Step 5: Generating constraints...")
            constraints = self._generate_constraints(
                problem_text, sets, parameters, variables
            )
            if not isinstance(constraints, list):
                result["errors"].append("Failed to generate constraints")
                return result
            result["constraints"] = constraints
            print(f"Generated {len(constraints)} constraints")

            print("Step 6: Building complete formulation...")
            formulation = self._build_formulation_str(
                sets=str(sets),
                parameters=str(parameters),
                decision_variables=str(variables),
                objective=str(objective),
                constraints=str(constraints),
            )
            result["formulation"] = formulation
            print("Built complete formulation")

            print("Step 7: Generating Gurobipy code...")
            code_response = self._generate_code(formulation)
            result["code"] = code_response
            if not result["code"]:
                result["errors"].append("Failed to generate code")
                return result
            result["success"] = True

        except Exception as e:
            result["errors"].append(f"Unexpected error in forward function: {str(e)}")
            print(f"Error in forward function: {e}")

        return result
