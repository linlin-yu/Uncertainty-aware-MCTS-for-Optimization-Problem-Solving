import math
from utils.utility_functions import (
    process_generated_assumptions,
    is_valid_state,
    semantic_cluster_states,
)
from feedback import evaluate_formulation
from node import FormulationNode, get_key_by_level
from mcts_class import MCTS


class EMCTS(MCTS):
    """Monte Carlo Tree Search for optimization formulation generation with dynamic component generation"""

    # Class constants
    LEVEL_ROOT, LEVEL_TYPE, LEVEL_INSTRUCTION = 0, 1, 2
    LEVEL_SETPARAM, LEVEL_MODEL, LEVEL_CODE = 3, 4, 5
    SOLVER_LIST = ["gurobipy", "pyomo_cbc", "pyomo_scip"]
    INDENT = "                    "

    def __init__(
        self,
        problem_description,
        llm_component_generator,
        llm_formulation_evaluator,
        exploration_weight,
        gamma,
        max_retry_time,
    ):
        self.problem_description = problem_description
        self.llm_component_generator = llm_component_generator
        self.llm_formulation_evaluator = llm_formulation_evaluator
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.max_retry_time = max_retry_time

        self.root = FormulationNode(problem_text=problem_description)
        self.root.visits = 1

    def _expand_node(self, node, num_components=None):
        """Expand node at mathematical formulation level."""
        if node.level + 1 == self.LEVEL_CODE:
            self._expand_code(node)
        else:
            self._expand_formulation(node, num_components)

    def _expand_formulation(self, node, num_components):
        """Expand node with formulation components."""
        print(
            f"[Level - {node.level + 1}: State Generation]: Generating {num_components} candidates"
        )

        candidate_states, state_entropies, state_log_probabilities = [], [], []

        for i in range(num_components):
            generated_formulation, formulation_entropy, logprob = (
                self.llm_component_generator.generate_next_component(
                    trajectory=node.trajectory, level=node.level + 1
                )
            )

            if is_valid_state(generated_formulation):
                candidate_states.append(generated_formulation)
                state_entropies.append(formulation_entropy)
                state_log_probabilities.append(logprob)
            else:
                print(f"Warning - Empty state generated at iteration {i + 1}")

        unique_states, semantic_uncertainty = semantic_cluster_states(
            candidate_states=candidate_states,
            problem_description=self.problem_description,
            state_entropies=state_entropies,
            state_log_probabilities=state_log_probabilities,
            check_entailment_function=self.llm_component_generator.check_entailment,
            ablation_study=True,
            indent=self.INDENT,
        )

        if unique_states:
            node.available_next_states.extend(unique_states)
            node.has_been_expanded = True
            node.uncertainty = semantic_uncertainty

            for unique_state in unique_states:
                print("Math Formulation:")
                print(unique_state["component"])

    def _expand_code(self, node):
        """Expand node at code level by generating solver code components."""
        print(
            f"[Level - {node.level}: State Generation]: Generating {len(self.SOLVER_LIST)} candidates"
        )

        states_generated = 0
        for solver_type in self.SOLVER_LIST:
            generated_code, code_entropy, _ = (
                self.llm_component_generator.generate_next_component(
                    trajectory=node.trajectory,
                    level=self.LEVEL_CODE,
                    solver_type=solver_type,
                )
            )

            if generated_code is not None:
                state_info = {
                    "component": generated_code,
                    "prob": 1.0 / len(self.SOLVER_LIST),
                    "mean_entropy": code_entropy,
                    "variance_entropy": 0.0,
                    "solver": solver_type,
                }
                node.available_next_states.append(state_info)
                states_generated += 1

                if solver_type == self.SOLVER_LIST[0]:
                    print("Gurobipy code")
                    print(generated_code)
            else:
                print(f"Warning: Empty state generated for solver {solver_type}")

        if states_generated > 0:
            node.has_been_expanded = True

    def _revise_node(self, current_node, previous_generation, error_message):
        """Generate revised structured formulation component."""
        print(
            f"[Level - {current_node.level}: State Revision]: Revising structured formulation"
        )

        generated_state, state_entropy, _ = (
            self.llm_component_generator.generate_next_component(
                trajectory=current_node.trajectory,
                level=current_node.level,
                previous_generation=previous_generation,
                error_message=error_message,
                solver_type=getattr(current_node, "solver_type", None),
            )
        )

        """Common logic for handling revision results."""
        if generated_state is None:
            print(
                f"[Level - {current_node.level}: State Revision]: Warning - Failed to generate revised state"
            )
            return False

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

        """Try to create a child node after successful revision."""
        parent = current_node.parent

        if (
            len(parent.children) < current_node.max_children
            and parent.available_next_states
        ):
            return self._create_child(parent)
        else:
            print("   Parent at capacity, skipping iteration")
            if hasattr(current_node, "error_message"):
                current_node.error_message = []
            return None

    def _create_child(self, current_node):
        """Create a child node by selecting an unexpanded component."""
        selected_next_state = current_node.available_next_states.pop(0)
        next_level = current_node.level + 1

        child = FormulationNode(
            parent=current_node, level=next_level, component_info=selected_next_state
        )
        current_node.children.append(child)
        return child

    def _handle_node_expansion(self, node, num_components):
        """Handle node expansion logic for both healthy and error nodes."""
        if not node.need_revision:
            """Handle expansion of healthy nodes."""
            if not node.has_been_expanded:
                print("Expanding node with new components at level", node.level)
                self._expand_node(node, num_components)

            return self._create_child(node) if node.available_next_states else node
        else:
            """Handle revision of nodes with errors."""
            print("Expanding node with error revision at level", node.level)

            if not hasattr(node, "error_message") or not node.error_message:
                print("Warning: Node marked for revision but no error message found")
                return None

            error_message = (
                node.error_message[-1]
                if isinstance(node.error_message, list)
                else node.error_message
            )

            return self._revise_node(
                node, previous_generation=node.component, error_message=error_message
            )

    def _backpropagate(self, node, rollout_value, rollout_variance, reasoning, gamma):
        """
        Update statistics for all nodes in the path from the given node to root.

        Performs value backpropagation using temporal difference learning with:
        - Incremental mean updates for Q-values
        - Variance tracking for uncertainty estimation
        - Error message propagation for node revision

        Parameters:
        -----------
        node : FormulationNode
            The leaf node to start backpropagation from
        rollout_value : float
            The evaluation reward for the current formulation
        rollout_variance : float
            The variance of the rollout evaluation
        reasoning : dict
            LLM feedback containing error messages by formulation level
        gamma : float
            Gamma parameter for temporal discounting (0 < gamma <= 1)
        """

        # Initialize leaf node with rollout results
        node.state_value = rollout_value
        node.state_variance = rollout_variance

        # Propagate values up the tree
        current_node = node
        while current_node is not None:
            # Calculate discounted return: nu = reward + gamma * value
            discounted_return = current_node.reward + gamma * current_node.state_value
            discounted_variance = (
                current_node.reward_variance + gamma**2 * current_node.state_variance
            )

            # Update visit count
            current_node.visits += 1

            # Incremental update of Q-value (running average)
            q_delta = discounted_return - current_node.q_value
            current_node.q_value += q_delta / current_node.visits

            # Incremental update of Q-variance using standard deviation
            # Formula: sqrt(var_new) = sqrt(var_old) + (sqrt(var_sample) - sqrt(var_old)) / n
            old_q_std = math.sqrt(
                max(0, current_node.q_variance)
            )  # Ensure non-negative
            sample_std = math.sqrt(max(0, discounted_variance))
            std_delta = (sample_std - old_q_std) / current_node.visits
            new_q_std = old_q_std + std_delta
            current_node.q_variance = max(0, new_q_std**2)  # Convert back to variance

            # Process error feedback from LLM reasoning
            if current_node.level in [
                self.LEVEL_CODE,
                self.LEVEL_MODEL
            ]:
                formulation_key = get_key_by_level(current_node.level)
                if reasoning[formulation_key]:
                    current_node.error_message.append(reasoning[formulation_key])
                    current_node.need_revision = True

            # Move to parent node
            current_node = current_node.parent

            # Update parent's state value based on children (if parent exists)
            if current_node is not None:
                current_node.update_state_value()

    def _select(self):
        """
        Select a node to expand using UCT

        Returns:
        --------
        FormulationNode
            The selected node
        """
        node = self.root

        # Traverse the tree until we reach a leaf node or non-fully expanded node
        while not node.is_terminal() and node.is_fully_expanded():
            # Find child with highest UCT score
            best_child = None
            best_score = float("-inf")

            for child in node.children:
                uct_score = child.get_puct_score(node.visits, self.exploration_weight)
                if uct_score > best_score:
                    best_score = uct_score
                    best_child = child

            if best_child is None:
                break

            node = best_child

        return node

    def search(self, iterations=None, num_components=None):
        """
        Perform MCTS search for a specified number of iterations.

        Parameters:
        -----------
        iterations : int
            Maximum number of iterations to perform
        num_components : int
            Number of components to consider during expansion

        Returns:
        --------
        tuple
            (best_formulation, total_evaluations, best_formulation_index)
        """
        # Initialize search tracking variables
        evaluation_count = 0
        best_reward = float("-inf")
        best_solution_index = None
        best_solution_obj = None
        freeze_cout = 0

        print("=" * 50 + " MCTS SEARCH STARTED " + "=" * 50)

        while evaluation_count < iterations:
            # SELECTION: Navigate from root to leaf or not fully expanded node using UCT scores
            print("\n" + "=" * 50)
            print("Select Node")
            print("=" * 50)
            selected_node = self._select()
            print(
                f"Selected Node: Level {selected_node.level} | "
                # f"Children: {len(selected_node.children)} | "
                # f"Available States: {len(selected_node.available_next_states)} | "
                # f"Fully Expanded: {selected_node.is_fully_expanded()}"
            )

            # EXPANSION: Create new nodes based on current state
            expanded_node = self._handle_node_expansion(selected_node, num_components)

            # Skip iteration if expansion failed
            if expanded_node is None:
                continue

            # SIMULATION: SKIP
            # formulation = self._simulate(node)

            # EVALUATION: Process terminal nodes, which contains the solver code, we need to execute it and verify the feasibility
            if expanded_node.is_terminal():
                if not expanded_node.has_been_evaluated:
                    freeze_cout = 0
                    evaluation_count += 1
                    print("\n" + "=" * 50)
                    print(f"Evaluation #{evaluation_count}")
                    print("=" * 50)
                    # Evaluate the formulation
                    # (
                    #     rollout_value,
                    #     rollout_variance,
                    #     backward_reasoning,
                    #     optimal_objective,
                    #     exe_success,
                    # ) = evaluate_formulation(
                    #     problem_description=self.problem_description,
                    #     forward_instruction=self.root.component,
                    #     assumption=expanded_node.trajectory["assumption"],
                    #     math_formulation=expanded_node.trajectory["formulation"],
                    #     code=expanded_node.trajectory["solver_code"],
                    #     llm_evaluator=self.llm_formulation_evaluator,
                    #     solver_type=expanded_node.solver,
                    # )
                    (
                        rollout_value,
                        rollout_variance,
                        backward_reasoning,
                        optimal_objective,
                        exe_success,
                    ) = evaluate_formulation(
                        problem_description=self.problem_description,
                        code=expanded_node.trajectory["solver_code"],
                        llm_evaluator=self.llm_formulation_evaluator,
                        solver_type=expanded_node.solver,
                    )
                    # The leaf node should log the information
                    expanded_node.has_been_evaluated = True
                    expanded_node.trajectory["optimal_objective"] = optimal_objective
                    expanded_node.evaluation_index = evaluation_count
                    if exe_success is None:
                        expanded_node.parent.code_fail_times += 1
                        if expanded_node.parent.code_fail_times >= self.max_retry_time:
                            error_message = []
                            for child in expanded_node.parent.children:
                                if isinstance(error_message, list):
                                    error_message.extend(child.error_message)
                            expanded_node.need_revision = False
                            expanded_node.parent.need_revision = True
                            expanded_node.parent.error_message.extend(error_message)
                            expanded_node.parent.error_message.append(
                                "Code execution failed after maximum retries"
                            )
                            expanded_node.parent.available_next_states = []
                            continue
                    # BACKPROPAGATION: Update node values up the tree
                    print("\n" + "=" * 50)
                    print("Backprogation")
                    print("=" * 50)
                    self._backpropagate(
                        expanded_node,
                        rollout_value,
                        rollout_variance,
                        backward_reasoning,
                        self.gamma,
                    )

                    # Track best performing formulation
                    if rollout_value > best_reward:
                        best_reward = rollout_value
                        best_solution_index = evaluation_count
                        best_solution_obj = optimal_objective
                        best_solution_exe_output = exe_success
                        print(
                            f"NEW BEST! Reward: {best_reward:.4f} at evaluation #{best_solution_index} with objective {best_solution_obj}"
                        )

                    # Early stopping: Check if search space is exhausted
                    if self._all_nodes_fully_expanded(self.root):
                        print(
                            f"Search completed - all nodes explored at evaluation #{evaluation_count}"
                        )
                        break
                else:
                    freeze_cout += 1
                    print(
                        f"Node already evaluated at evaluation #{expanded_node.evaluation_index}, skipping"
                    )
                    if freeze_cout >= 3:
                        break

        # Finalize search results
        if best_solution_index is None:
            print("No valid formulation found, using best available node")

        print(
            "=" * 50
            + f" MCTS SEARCH COMPLETED (Evaluation {evaluation_count})"
            + "=" * 50
        )
        self.print_tree()
        # self.print_full_tree()
        return best_solution_exe_output, best_solution_obj
