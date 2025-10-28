import io
import re
import signal
from contextlib import redirect_stdout
from typing import Dict, Any
import gurobipy as gp
from gurobipy import GRB
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def evaluate_formulation(
    problem_description,
    code,
    llm_evaluator,
    solver_type,
):
    """
    Evaluate a mathematical formulation by generating code, executing it, and assessing results.

    Returns:
        tuple: (rollout_value, variance_rollout_value, combined_reasoning)
    """
    backward_reasoning = {
        "formulation": None,  # Level 2: formulation
        "code": None,  # Level 3: solver code
    }

    # ==================== CODE GENERATION & EXECUTION ====================
    print("\n" + "-" * 20 + " Code Execution " + "-" * 20)
    # Execute the code and capture results
    print(f"Detected solver: {solver_type}")
    execution_output = execute_optimization_code(code, solver_type)

    # Initialize evaluation metrics
    solver_score = 0
    backward_reasoning["code"] = execution_output.get("error_message", None)
    solver_confidence = 1.0

    # Check if execution was successful
    print(f"[Execution Success]: {execution_output['success']}")
    if not execution_output["success"]:
        print(execution_output["error_message"])
        solver_score = 0.0
        optimal_objective = None
        rollout_variance = 0.0
        return (
            solver_score,
            rollout_variance,
            backward_reasoning,
            optimal_objective,
            execution_output["success"],
        )

    # Check the alignment
    alignment_score, align_explain = llm_evaluator.alignment_check(
        problem_description, code
    )

    # Check the feasibility
    feasiblity_score = 0
    optimal_result = extract_optimization_results(
        execution_output["output"]
    )  # status, objective, variables, status_message
    optimal_objective = optimal_result.get("opt_obj")
    optimal_variables = optimal_result.get("opt_var")

    # Evaluate different solver outcomes
    if optimal_objective in ["unbounded", "unclear"]:
        print(f"Solver result: {optimal_objective.upper()}")
        feasiblity_explain = "The optimal objective is unclear"

    elif optimal_objective == "infeasible":
        print("Problem is identified as INFEASIBLE")
        feasiblity_explain = "Problem is identified as INFEASIBLE"

    elif optimal_objective and optimal_variables is not None:
        print(f"Optimal Objective: {optimal_objective}")
        print(f"Optimal Variables: {optimal_variables}")
        print("\n" + "-" * 20 + " Feasiblity Check " + "-" * 20)
        # Check solution feasibility with LLM
        feasiblity_score, feasibility_explain = llm_evaluator.alignment_check(
            problem_description, optimal_objective
        )
    else:
        feasiblity_score = 0
        feasiblity_explain = "EXECUTION EXCEPTION occurred"

    # Combine solver and formulation scores
    rollout_variance = 1 - solver_confidence

    print(f"Rollout Value: {solver_score}")
    print(f"Rollout Variance: {rollout_variance:.3f}")
    print(f"Backward Reasoning: {backward_reasoning}")

    return (
        solver_score,
        rollout_variance,
        backward_reasoning,
        optimal_objective,
        execution_output["output"],
    )


def detect_solver_from_code(code):
    """
    Detect the solver type from code content.

    Args:
        code (str): The code to analyze

    Returns:
        str: The detected solver name
    """
    solver_patterns = {
        "gurobipy": ["gurobipy"],
        "pyomo_cbc": ["cbc", "ipopt"],
        "pyomo_scip": ["scip"],
    }

    code_lower = code.lower()
    for solver, patterns in solver_patterns.items():
        if any(pattern in code_lower for pattern in patterns):
            return solver

    print(
        "[State Revision]: Warning - Undefined solver in code, defaulting to gurobipy"
    )
    return "gurobipy"


class TimeoutException(Exception):
    """Custom exception for timeout handling"""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Code execution timeout")


def execute_optimization_code(
    code: str, solver_type: str = "gurobipy", timeout: int = 30
) -> Dict[str, Any]:
    """
    Execute optimization code (Gurobipy or Pyomo) with timeout and comprehensive error handling.

    Args:
        code (str): The optimization code to execute
        solver_type (str): Type of solver - "gurobipy", "pyomo_cbc", "pyomo_ipopt", or "pyomo_scip"
        timeout (int): Timeout in seconds (default: 30)

    Returns:
        Dict[str, Any]: Execution results containing:
            - 'success' (bool): True if execution completed without errors
            - 'output' (str): Complete execution output
            - 'error_message' (str|None): Merged error information for debugging if failed, None if successful
    """
    # Initialize execution context
    output_buffer = io.StringIO()

    # Set up namespace based on solver type
    if solver_type == "gurobipy":
        namespace = {"gp": gp, "GRB": GRB}
    else:
        namespace = {"pyo": pyo, "SolverFactory": SolverFactory}
    # Initialize result structure
    result = {"success": False, "output": "", "error_message": None}

    # Set up timeout signal handler (only on Unix-like systems)
    old_handler = None
    timeout_supported = hasattr(signal, "SIGALRM")

    if timeout_supported:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    try:
        # Execute the code with output redirection
        with redirect_stdout(output_buffer):
            exec(code, namespace)

        # If we reach here, execution was successful
        result["success"] = True

    except TimeoutException:
        error_message = f"Error Type: TimeoutError\nError Message: Code execution timed out after {timeout} seconds"
        result["error_message"] = error_message
        output_buffer.write(f"\n{error_message}\n")

    except Exception as e:
        # Merge all exception information into error_message
        error_type = type(e).__name__
        error_str = str(e)

        error_message = f"Error Type: {error_type}\nError Message: {error_str}"
        result["error_message"] = error_message

        # Add error information to output buffer
        output_buffer.write(f"\n{error_message}")

    finally:
        # Clean up timeout signal (only if supported)
        if timeout_supported:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    # Capture the complete output
    result["output"] = output_buffer.getvalue()
    if result["output"] is None:
        result["success"] = False
        result["error_message"] = (
            "The code is considered as a string and can not be directly executed."
        )
    print("[Execution Output]:", result["output"])
    print("[Execution Error Message]:", result["error_message"])
    # print(code)
    return result


def extract_execution_output_gurobipy(execution_output):
    """
    Extract the optimal objective value and variable assignments from Gurobipy output.

    Assumes the output includes:
    - "Objective Value: <value>"
    - Variable values between "BEGIN_VARIABLES" and "END_VARIABLES"
      in the format: "Variable <name>: <value>"

    Parameters
    ----------
    execution_output : str
        Full raw text output from Gurobipy model execution.

    Returns
    -------
    tuple
        - opt_obj: str or None, the objective value (e.g., "90720.00000000")
        - opt_variables: str or None, multiline string of variable assignments
    """
    # Extract objective value
    obj_match = re.search(r"Objective Value:\s*([+-]?\d+(?:\.\d+)?)", execution_output)
    opt_obj = obj_match.group(1) if obj_match else None

    # Extract lines between BEGIN_VARIABLES and END_VARIABLES
    var_block_match = re.search(
        r"BEGIN_VARIABLES\s*(.*?)\s*END_VARIABLES", execution_output, re.DOTALL
    )
    opt_variables = var_block_match.group(1).strip() if var_block_match else None

    return opt_obj, opt_variables


def extract_optimization_results(execution_output: str) -> Dict[str, Any]:
    """
    Extract optimization results from execution output for different solvers.

    This function parses the execution output to determine the optimization status
    and extract objective values and variable assignments when available.

    Args:
        execution_output (str): Complete output from optimization code execution
        solver_type (str): Type of solver used - "gurobipy", "pyomo_cbc", "pyomo_ipopt", or "pyomo_scip"

    Returns:
        Dict[str, Any]: Parsed results containing:
            - 'status' (str): One of 'optimal', 'infeasible', 'unbounded', 'error', 'unclear'
            - 'objective_value' (str|None): Objective value if optimal solution found
            - 'variables' (str|None): Variable assignments if optimal solution found
            - 'status_message' (str): Human-readable status description
    """

    # Initialize result structure
    result = {
        "status": "unclear",
        "opt_obj": None,
        "opt_var": None,
        "status_message": "Status could not be determined",
    }

    # Convert to lowercase for case-insensitive matching
    execution_output_lower = execution_output.lower()

    # Check optimization status
    if "optimal solution found" in execution_output_lower:
        result["status"] = "optimal"
        opt_objective, opt_variables = extract_execution_output_gurobipy(
            execution_output
        )

    elif "infeasible" in execution_output_lower:
        result["status"] = "infeasible"
        opt_objective = "infeasible"
        opt_variables = None

    elif "unbounded" in execution_output_lower:
        result["status"] = "unbounded"
        opt_objective = "unbounded"
        opt_variables = None

    elif "timeout" in execution_output_lower:
        result["status"] = "timeout"
        opt_objective = None
        opt_variables = None

    else:
        result["status"] = "error"
        opt_objective = None
        opt_variables = None

    result["opt_obj"] = opt_objective
    result["opt_var"] = opt_variables
    # print(f"[Optimization Status]: optimal objective={result['opt_obj']}")
    # print(
    #     f"[Optimization Variables]: {result['opt_var'] if result['opt_var'] else None}"
    # )
    return result
