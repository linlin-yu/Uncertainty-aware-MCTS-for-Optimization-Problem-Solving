import math
from typing import Dict, Any, List, Tuple, Optional
from utils.semantic_utils import analyze_response_semantics

import os
import requests

# Import necessary libraries for text similarity
from difflib import SequenceMatcher


def filter_similar_formulations_strmatch(components, similarity_threshold):
    """
    Filter out components that are too similar to others

    Parameters:
    -----------
    components : list
        List of formulation components to filter
    similarity_threshold : float
        Threshold for determining if two formulations are too similar (0-1)

    Returns:
    --------
    list
        Filtered list of components with reduced redundancy
    """
    if not components:
        return []

    def similarity(a, b):
        """Calculate string similarity ratio between two formulations"""
        return SequenceMatcher(None, str(a), str(b)).ratio()

    # Keep track of components we want to keep
    unique_components = [components[0]]

    # Compare each component with the ones we've decided to keep
    for component in components[1:]:
        is_unique = True
        for unique_comp in unique_components:
            # Compare the formulation strings
            if similarity(component, unique_comp) > similarity_threshold:
                is_unique = False
                break

        if is_unique:
            unique_components.append(component)

    return unique_components

def refine_mamo_problem(problem_text, api_key=None):
    """
    Use an LLM refine the problem to make it correctly understand the "difference" in Mamo dataset.
    The phrase 'difference between X and Y' here should be interpreted as a simple subtraction (i.e., X - Y), rather than as an absolute value.

    Parameters:
    -----------
    problem_text : str
        Original problem
    api_key : str, optional
        OpenAI API key. If None, will try to use the key from environment variables.

    Returns:
    --------
    str
        Refined problem
    """

    # Use provided API key or get from environment
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("No API key available for LLM evaluation")
        return 0

    api_url = "https://api.openai.com/v1/chat/completions"
    model = "gpt-4o"

    prompt = f"""
    Given an optimization problem, I first need to verify and potentially correct certain aspects of the problem statement.

    Problem description:
    {problem_text}

    Normally, the phrase "difference between X and Y" should mean abs(X - Y).
    However, in this problem set, "difference between X and Y" actually refers to a simple subtraction: X - Y.
    For example, the statement like "Also, the difference in resource allocation between procurement ($X$) and distribution ($Z$) 
    should not exceed 200 units to maintain a balanced flow of goods through the supply chain" in the problem should be
    understanded as X - Y <= 200

    Your task is:
    Identify whether the phrase "difference between X and Y" or similar expressions appear in the problem.
    + If no such phrase appears, return the original problem unchanged.
    + If such a phrase does appear, modify only that sentence to explicitly mean X minus Y. Do not change any other parts of the problem.

    IMPORTANT: Do not alter any sentence that doesn't contain a "difference between X and Y"-like expression.

    Return your response in this JSON format:
    {{"description": "Your refined problem description"}}
    
    """

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,  # Lower temperature for more consistent evaluations
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()

        # Extract the content from the response
        response = response_json["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        response = ""
    except Exception as e:
        print(f"Unexpected error: {e}")
        response = ""

    clean_response = response
    if "```" in response:
        # Extract content between triple backticks
        first_marker = response.find("```")
        if first_marker >= 0:
            # Find the closing backticks
            second_marker = response.find("```", first_marker + 3)
            if second_marker > first_marker:
                # Extract the content and check if there's language identifier
                content = response[first_marker + 3 : second_marker].strip()
                if content.startswith("json"):
                    content = content[len("json") :].strip()
                if content.startswith("python"):
                    content = content[len("python") :].strip()
                clean_response = content

    try:
        # Try to evaluate the response as Python code
        result = eval(clean_response)
        return result if isinstance(result, dict) else {}
    except Exception as e:
        print(f"[Error parsing LLM response]: {e}")
        print(f"[Raw response]: {response}")
        print(f"[Cleaned response]: {clean_response}")
        return {}

def clean_json_mark_list(raw_response):
    """
    Clean and extract JSON list from raw response text.

    This function removes markdown formatting and extracts the JSON array
    from a raw response string, handling various formatting patterns.

    Parameters
    ----------
    raw_response : str or any
        Raw response that may contain JSON list with markdown formatting

    Returns
    -------
    str
        Cleaned JSON list string, or empty string if no valid JSON found
    """
    # Convert to string and clean whitespace
    text = str(raw_response).strip()
    if not text:
        return []
    if isinstance(text, list):
        return text
    elif isinstance(text, str):
        # Remove markdown code blocks
        text = remove_markdown_blocks(text)

        # Extract JSON array boundaries
        json_text = extract_json_array(text)
        return json_text
    else:
        return raw_response


def remove_markdown_blocks(text):
    """
    Remove markdown code block formatting from text.

    Parameters
    ----------
    text : str
        Input text that may contain markdown formatting

    Returns
    -------
    str
        Text with markdown formatting removed
    """
    # Handle specific JSON markdown blocks
    if "```json" in text:
        start_marker = "```json"
        start_idx = text.find(start_marker) + len(start_marker)
        end_idx = text.find("```", start_idx)

        if end_idx != -1:
            return text[start_idx:end_idx].strip()
        else:
            # If closing ``` not found, take everything after ```json
            return text[start_idx:].strip()

    # Handle generic code blocks
    elif text.startswith("```") and text.endswith("```"):
        return text[3:-3].strip()

    return text


def extract_json_array(text):
    """
    Extract JSON array from text by finding bracket boundaries.

    Parameters
    ----------
    text : str
        Input text containing JSON array

    Returns
    -------
    str
        Extracted JSON array string, or original text if no valid array found
    """
    first_bracket = text.find("[")
    last_bracket = text.rfind("]")

    # Check if we have valid bracket pair
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        return text[first_bracket : last_bracket + 1]

    # Return original text if no valid JSON array structure found
    return text


def process_generated_assumptions(generated_assumptions) -> List[str]:
    """
    Process generated assumptions into a consistent list format.

    Args:
        generated_assumptions: Can be None, str, or list of strings

    Returns:
        List of assumption strings, with default fallback if input is invalid
    """
    if not generated_assumptions:
        return ["The problem statement is clear and unambiguous."]

    if isinstance(generated_assumptions, str):
        return [generated_assumptions]
    elif isinstance(generated_assumptions, list):
        return generated_assumptions
    else:
        return ["The problem statement is clear and unambiguous."]


def is_valid_state(state) -> bool:
    """
    Check if generated state is valid.

    Args:
        state: The state to validate (can be any type)

    Returns:
        True if state is not None and has non-empty string representation
    """
    return state is not None and len(str(state).strip()) > 0


def create_assumption_state(
    combination: Dict[str, Any], probability: float
) -> Dict[str, Any]:
    """
    Create a state dictionary for an assumption combination.

    Args:
        combination: Dictionary mapping aspects to chosen assumptions
        probability: Probability of this combination

    Returns:
        Dictionary representing the state with component, probability, and entropy info
    """
    mean_entropy = -probability * math.log(probability) if probability > 0 else 0.0

    return {
        "component": combination,
        "prob": probability,
        "mean_entropy": mean_entropy,
        "variance_entropy": 0.0,
    }


def run_ablation_study(
    candidate_states: List[str], unique_states_with_stats: List[Dict]
) -> None:
    """
    Run ablation study comparing semantic vs string matching clustering.

    Args:
        candidate_states: Original list of candidate states
        unique_states_with_stats: Results from semantic clustering

    Note:
        Requires filter_similar_formulations_strmatch function to be available
    """

    string_match_results = filter_similar_formulations_strmatch(candidate_states, 0.8)
    if len(unique_states_with_stats) != len(string_match_results):
        print("[Ablation on the semantic similarity clustering]:")
        print(
            f"With semantic similarity, we have {len(unique_states_with_stats)} unique components."
        )
        print(
            f"With string match, we have {len(string_match_results)} unique components."
        )


def semantic_cluster_states(
    candidate_states: List[str],
    problem_description: str,
    state_entropies: Optional[List[float]] = None,
    state_log_probabilities: Optional[List[float]] = None,
    check_entailment_function=None,
    ablation_study: bool = False,
    indent: str = "                    ",
) -> Tuple[List[Dict], float]:
    """
    Cluster semantically similar states and return processed states.

    Args:
        candidate_states: List of generated states to cluster
        problem_description: The problem description for context
        state_entropies: Optional list of entropy values for each state
        state_log_probabilities: Optional list of log probabilities for each state
        check_entailment_function: Function to check semantic entailment
        ablation_study: Whether to run ablation study comparison
        indent: Indentation string for print statements

    Returns:
        Tuple of (clustered_states, semantic_uncertainty)

    Note:
        Requires analyze_response_semantics function to be available
    """
    if not candidate_states:
        print(f"{indent}[State Generation]: Failed to generate any valid states")
        return [], 0.0

    # Preprocess inputs
    if state_entropies is None:
        state_entropies = [0.0] * len(candidate_states)
    if state_log_probabilities is None:
        state_log_probabilities = [0.0] * len(candidate_states)

    # Cluster responses semantically
    unique_states_with_stats, semantic_uncertainty = analyze_response_semantics(
        responses=candidate_states,
        question=problem_description,
        response_logprobs=state_log_probabilities,
        response_entropies=state_entropies,
        check_entailment_function=check_entailment_function,
    )

    # Optional ablation study
    if ablation_study:
        run_ablation_study(candidate_states, unique_states_with_stats)

    # Sort by decreasing probability order
    unique_states_with_stats.sort(key=lambda x: x["prob"], reverse=True)

    print(
        f"{indent}[State Generation]: Successfully generated {len(unique_states_with_stats)} unique states"
    )

    return unique_states_with_stats, semantic_uncertainty


def handle_revision_result(current_node, generated_state, state_entropy) -> bool:
    """
    Common logic for handling revision results.

    Args:
        current_node: The node being revised
        generated_state: The newly generated state (could be None)
        state_entropy: Entropy of the generated state

    Returns:
        True if revision was successful, False otherwise
    """
    if generated_state is not None:
        revised_state = {
            "component": generated_state,
            "prob": current_node.probability * 0.9,
            "mean_entropy": state_entropy,
            "variance_entropy": 0.0,
        }

        # Add solver info for code revisions
        if hasattr(current_node, "solver"):
            revised_state["solver"] = current_node.solver

        current_node.parent.available_next_states.insert(0, revised_state)
        current_node.need_revision = False
        current_node.probability *= 0.1

        print(
            f"[Level - {current_node.level}: State Revision]: Successfully revised component"
        )
        return True
    else:
        print(
            f"[Level - {current_node.level}: State Revision]: Warning - Failed to generate revised state"
        )
        return False


def validate_node_attributes(node, required_attributes: List[str]) -> bool:
    """
    Validate that a node has all required attributes.

    Args:
        node: The node to validate
        required_attributes: List of attribute names that must be present

    Returns:
        True if all attributes are present, False otherwise
    """
    for attr in required_attributes:
        if not hasattr(node, attr):
            print(f"Warning: Node missing required attribute '{attr}'")
            return False
    return True


def safe_get_trajectory_value(node, key: str, default: str = "") -> str:
    """
    Safely get a value from node's trajectory dictionary.

    Args:
        node: The node containing trajectory information
        key: The key to look up in the trajectory
        default: Default value if key is not found

    Returns:
        The trajectory value or default
    """
    if not hasattr(node, "trajectory") or not isinstance(node.trajectory, dict):
        return default
    return node.trajectory.get(key, default)


def calculate_state_entropy(probability: float) -> float:
    """
    Calculate entropy for a given probability.

    Args:
        probability: The probability value (0 < p <= 1)

    Returns:
        Entropy value (-p * log(p))
    """
    if probability <= 0:
        return 0.0
    return -probability * math.log(probability)


def format_level_message(level: int, message: str) -> str:
    """
    Format a message with level information.

    Args:
        level: The level number
        message: The message to format

    Returns:
        Formatted message string
    """
    return f"[Level - {level}]: {message}"
