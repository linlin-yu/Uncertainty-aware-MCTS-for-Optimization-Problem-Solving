import io
import re
import signal
import traceback
from contextlib import redirect_stdout
from typing import Dict, Any
import gurobipy as gp
from gurobipy import GRB


class TimeoutException(Exception):
    """Custom exception for timeout handling"""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Code execution timeout")


def exec_gurobipy_code(gurobi_code, output_buffer, timeout=30):
    """
    Execute Gurobi Python code with timeout and error handling.

    Args:
        gurobi_code (str): The Gurobi Python code to execute
        output_buffer (StringIO): Buffer to capture output
        timeout (int): Timeout in seconds (default: 30)

    Returns:
        str: Execution output (same as original function)
    """
    namespace = {
        "gp": gp,
        "GRB": GRB,
    }

    # Set up timeout signal handler (only on Unix-like systems)
    old_handler = None
    timeout_supported = hasattr(signal, "SIGALRM")
    is_executable = True
    error_msg = None

    if timeout_supported:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

    try:
        with redirect_stdout(output_buffer):
            exec(gurobi_code, namespace)

    except TimeoutException:
        # Write timeout error to buffer
        is_executable = False
        output_buffer.write(
            f"Error: Code execution timed out after {timeout} seconds\n"
        )

    except Exception as e:
        is_executable = False
        # Capture any other errors and add to output buffer
        error_type = type(e).__name__
        error_msg = str(e)

        # Add error information to output buffer (don't clear existing output)
        output_buffer.write(f"\nError: {error_type}: {error_msg}\n")

        # Optionally add traceback for debugging
        if hasattr(e, "__traceback__"):
            traceback_lines = traceback.format_exception(type(e), e, e.__traceback__)
            output_buffer.write("Traceback:\n")
            for line in traceback_lines:
                output_buffer.write(line)

    finally:
        # Clean up timeout signal (only if supported)
        if timeout_supported:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    execution_output = output_buffer.getvalue()

    return execution_output, is_executable, error_msg


def generate_execute_gurobipy_with_retry(
    formulation: Any, code_generator: Any, max_retry_time: int, log_code_file: str
) -> Dict[str, Any]:
    """
    Generate and execute Gurobipy code with automatic retry on failure.

    This function attempts to generate Gurobipy code from a formulation,
    execute it, and extract the solution. If execution fails, it retries code generation
    and execution up to `max_retry_time` times using error feedback.

    Args:
        formulation: A structured representation of the optimization problem
        code_generator: Object that can generate Gurobi code from formulation
        max_retry_time: Maximum number of retries if code execution fails
        log_code_file: Path to file for logging execution details

    Returns:
        Dictionary containing:
            - code: Final Gurobipy code string, or None if failed
            - opt_obj: Extracted optimal objective value, or status string
            - opt_variables: Extracted optimal variable assignments, or None
            - cnt_exe_error: Number of failed execution attempts
            - exe_error: List of error messages from failed executions
            - execution_output: Final execution output
            - success: Boolean indicating overall success
    """
    # Initialize tracking variables
    log_messages = []
    cnt_exe_error = 0
    exe_error = []
    success = False
    code = None
    error_msg = None
    execution_output = ""

    def log_message(message: str) -> None:
        """Helper function to collect log messages."""
        log_messages.append(message)

    while cnt_exe_error <= max_retry_time and not success:
        log_message(
            f"[Gurobipy code generation attempt {cnt_exe_error + 1}]" + "-" * 40
        )
        code = code_generator.generate_gurobi_code(formulation, code, error_msg)
        if not code:
            log_message("[Gurobi code generation failed: Empty code returned]")
            cnt_exe_error += 1
            exe_error.append("Code generation returned empty code")
            continue

        # Log generated code
        log_message("[Generated Code]:")
        log_message(code)

        # Execute code
        output_buffer = io.StringIO()
        try:
            execution_output, is_executable, error_msg = exec_gurobipy_code(
                code, output_buffer
            )

            # Log execution results
            log_message("\n[Execution Output]:")
            log_message(execution_output)

            if is_executable:
                success = True
                log_message("[Execution Status]: SUCCESS")
            else:
                cnt_exe_error += 1
                exe_error.append(error_msg)
                log_message(f"[Execution Status]: FAILED - {error_msg}")

        except Exception as e:
            cnt_exe_error += 1
            error_msg = f"Execution exception: {str(e)}"
            exe_error.append(error_msg)
            log_message(f"[Execution Status]: EXCEPTION - {error_msg}")

    # Process final results
    if success:
        log_message("[Gurobi code final optimization status]")
        execution_lower = execution_output.lower()

        if "optimal solution found" in execution_lower:
            log_message("Model solved and found OPTIMAL solution.")
            opt_obj, opt_variables = extract_execution_output_gurobipy(execution_output)
        elif "infeasible" in execution_lower:
            log_message("Model solved but found INFEASIBLE.")
            opt_obj = "infeasible"
            opt_variables = None
            exe_error.append("Model is infeasible")
        elif "unbounded" in execution_lower:
            log_message("Model solved but found UNBOUNDED.")
            opt_obj = "unbounded"
            opt_variables = None
            exe_error.append("Model is unbounded")
        else:
            log_message("Model solved but status unclear.")
            opt_obj = "unclear"
            opt_variables = None
            exe_error.append("Solver status unclear")
    else:
        final_message = f"Maximum retries ({max_retry_time}) reached."
        if error_msg:
            final_message += f" Last error: {error_msg}"
        log_message(f"[FINAL STATUS]: FAILED - {final_message}")
        opt_obj = None
        opt_variables = None

    # Prepare execution summary
    execution_summary = {
        "code": code,
        "opt_obj": opt_obj,
        "opt_variables": opt_variables,
        "cnt_exe_error": cnt_exe_error,
        "exe_error": exe_error,
        "execution_output": execution_output,
        "success": success
    }

    # Write all log messages to file at once - this ALWAYS executes
    with open(log_code_file, "a", encoding="utf-8") as log_file:
        log_file.write("\n".join(log_messages) + "\n")

    return execution_summary


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
