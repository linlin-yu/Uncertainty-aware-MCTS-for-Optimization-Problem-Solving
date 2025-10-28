import math
import copy

# Level constants
LEVEL_ROOT, LEVEL_TYPE, LEVEL_INSTRUCTION = 0, 1, 2
LEVEL_SETPARAM, LEVEL_MODEL, LEVEL_CODE = 3, 4, 5

# Updated mapping to align with actual levels and trajectory structure
LEVEL_KEY_MAPPING = {
    LEVEL_ROOT: "problem_description",
    LEVEL_TYPE: "type",
    LEVEL_INSTRUCTION: "instruction",
    LEVEL_SETPARAM: "setparams",
    LEVEL_MODEL: "formulation",
    LEVEL_CODE: "code",
}


def get_key_by_level(level):
    """Get the corresponding trajectory key for a given level."""
    return LEVEL_KEY_MAPPING.get(level)


class FormulationNode:
    """
    Node in the MCTS tree representing a partial or complete optimization formulation.

    Each node corresponds to a specific level in the formulation hierarchy:
    Level 0: Root node
    Level 1: Problem assumptions
    Level 2: formulation
    Level 3: Corresponding code (gurobipy, pyomo-glpk, pyomo-scip)
    """

    # Reward calculation constants
    ENTROPY_REWARD_MULTIPLIER = 10
    ENTROPY_VARIANCE_MULTIPLIER = 100

    def __init__(
        self,
        parent=None,
        level=0,
        component_info=None,
        max_children=6,
        problem_text=None,
    ):
        """
        Initialize a formulation node.

        Args:
            parent: Parent node in the MCTS tree
            level: Formulation level (0-6)
            component_info: Dictionary containing component, probability, and entropy info
            max_children: Maximum number of child nodes allowed
        """
        # Tree structure
        self.parent = parent
        self.children = []
        self.level = level
        self.max_children = max_children

        # MCTS statistics
        self.visits = 0
        self.reward = 0.0  # state-action reward, will be the -entropy
        self.reward_variance = 0.0  # Variance of state-action reward
        self.q_value = 0.0  # Action-value (Q)
        self.q_variance = 0.0  # Variance of Q-value
        self.state_value = 0.0  # State value (V)
        self.state_variance = 0.0  # Variance of state value
        self.uncertainty = 0.0  # Information-theoretic uncertainty

        # Node state tracking
        self.has_been_expanded = False
        self.has_been_evaluated = (
            False  # only effective for leaf node with complete formulation
        )

        self.available_next_states = []
        self.error_message = []
        self.need_revision = False
        self.exe_output = None

        # for the formulation node
        self.code_fail_times = 0
        # Node evaluation (only for the leaf node)
        self.optimal_objective = None  # Optimal objective value from solver execution
        self.evaluation_index = None  # Index of the evaluation in the search process
        self.solver = None  # Solver type for this node, to be set later
        # Initialize formulation based on node level
        if level == LEVEL_ROOT:
            self._initialize_root_node(problem_text)
        else:
            self._initialize_child_node(parent, component_info)

    def _initialize_root_node(self, problem_text):
        """Initialize the root node with empty formulation structure."""
        self.probability = 1.0
        self.trajectory = {
            "problem_description": problem_text,
            "instruction": None,
            "setparams": None,
            "formulation": None,
            "code": None,
            "optimal_objective": None,
        }

    def _initialize_child_node(self, parent, component_info):
        """Initialize a child node with component information."""
        if component_info is None or parent is None:
            raise ValueError("Child nodes require both parent and component_info")

        # Extract component information
        self.component = component_info["component"]
        self.probability = component_info["prob"]

        # Calculate reward based on entropy (lower entropy = higher reward)
        mean_entropy = component_info.get("mean_entropy", 0.1)
        variance_entropy = component_info.get("variance_entropy", 0.01)
        self.reward = 1 - mean_entropy * self.ENTROPY_REWARD_MULTIPLIER
        self.reward_variance = variance_entropy * self.ENTROPY_VARIANCE_MULTIPLIER
        if "solver" in component_info:
            self.solver = component_info["solver"]

        # Copy parent's formulation and add new component
        self.trajectory = copy.deepcopy(parent.trajectory)
        key = get_key_by_level(self.level)
        self.trajectory[key] = component_info["component"]

    def update_state_value(self):
        """
        Update the state value (V) based on children's Q-values and probabilities.
        Also calculates uncertainty using information entropy.
        """
        if not self.children:
            raise ValueError("Current node do not have childer.")

        # Calculate expected value and uncertainty
        q_values = []
        self.state_value = 0.0
        self.uncertainty = 0.0

        for child in self.children:
            # Expected value calculation
            self.state_value += child.probability * child.q_value
            q_values.append(child.q_value)

            # Information entropy calculation (higher entropy = more uncertainty)
            if child.probability > 0:
                self.uncertainty -= child.probability * math.log(child.probability)

        # Calculate variance of Q-values
        self.state_variance = self._calculate_variance(q_values)

    def _calculate_variance(self, values):
        """Calculate variance of a list of values."""
        if len(values) <= 1:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def is_fully_expanded(self):
        """
        Check if all possible children have been created.

        Returns:
            bool: True if node is terminal or all available states have been expanded
        """
        if self.need_revision:
            return False
        else:
            if self.is_terminal():
                return True
            # Node is fully expanded if it has been expanded and no more states are available
            return self.has_been_expanded and len(self.available_next_states) == 0

    def is_terminal(self):
        """
        Check if this node represents a complete formulation.

        A complete formulation must be at the constraints level (6) and contain:
        - Problem assumptions
        - At least one set, parameter, variable, and constraint
        - An objective function

        Returns:
            bool: True if this is a terminal node with complete formulation
        """
        if self.level == LEVEL_CODE:
            return True
        else:
            return False

    def get_uct_score(self, parent_visits, exploration_weight=2.0):
        """
        Calculate the Upper Confidence Bound for Trees (UCT) score.

        Args:
            parent_visits: Number of times parent node has been visited
            exploration_weight: Exploration vs exploitation trade-off parameter

        Returns:
            float: UCT score for node selection
        """
        if self.visits == 0:
            return float("inf")  # Prioritize unvisited nodes

        # Exploitation term: average reward
        exploitation = self.q_value

        # Exploration term: confidence interval
        exploration = exploration_weight * math.sqrt(
            math.log(parent_visits) / self.visits
        )

        return exploitation + exploration

    def get_puct_score(self, parent_visits, exploration_weight):
        """
        Calculate the Polynomial Upper Confidence Trees (PUCT) score.
        Incorporates prior probability in the exploration term.

        Args:
            parent_visits: Number of times parent node has been visited
            exploration_weight: Exploration vs exploitation trade-off parameter

        Returns:
            float: PUCT score for node selection
        """
        exploitation = self.q_value

        # Exploration term weighted by prior probability
        exploration_term = (
            self.probability * math.sqrt(parent_visits) / (1 + self.visits)
        )
        exploration = exploration_weight * exploration_term

        return exploitation + exploration
