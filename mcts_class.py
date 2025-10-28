from node import FormulationNode


class MCTS:
    """
    Monte Carlo Tree Search for optimization formulation generation with dynamic component generation
    """

    def __init__(
        self,
        problem_description,
        llm_component_generator,
        llm_formulation_evaluator,
        exploration_weight,
        gamma,
        max_retry_time,
        log_dir,
    ):
        self.problem_description = problem_description

        # LLM parts
        self.llm_component_generator = llm_component_generator
        self.llm_formulation_evaluator = llm_formulation_evaluator

        # Hyperparameters
        self.exploration_weight = exploration_weight
        self.gamma = gamma
        self.max_retry_time = max_retry_time  # Maximum retry time for generating code

        # initialize the root node of the MCTS tree
        self.root = FormulationNode()
        self.root.visits = 1
        self.log_dir = log_dir

    def print_tree(self):
        """
        Print the tree starting from the root node.
        """
        self._print_node(self.root, "", -1)

    def _print_node(self, node, prefix, index):
        """
        Recursively print the tree with node number and its level.
        """
        if index == -1:
            print(
                f"Root(visits={node.visits}, prob={node.probability:.1f}, reward={node.reward:.1f}, var_reward={node.reward_variance:.1f}, value={node.state_value:.1f}, var_value={node.state_variance:.1f}, q={node.q_value:.1f}, var_q={node.q_variance:.1f}, uncertainty={node.uncertainty:.1f})"
            )
        else:
            # Print the current node
            if node.error_message:
                print(
                    prefix
                    + f"Node-{node.level}-{index}(visits={node.visits}, prob={node.probability:.1f}, reward={node.reward:.1f}, var_reward={node.reward_variance:.1f}, value={node.state_value:.1f}, var_value={node.state_variance:.1f}, q={node.q_value:.1f}, var_q={node.q_variance:.1f}, uncertainty={node.uncertainty:.1f}, PUCT={node.get_puct_score(node.parent.visits, self.exploration_weight):.1f}, Error: {node.error_message})"
                )
            else:
                if node.optimal_objective:
                    print(
                        prefix
                        + f"Node-{node.level}-{index}(visits={node.visits}, prob={node.probability:.1f}, reward={node.reward:.1f}, var_reward={node.reward_variance:.1f}, value={node.state_value:.1f}, var_value={node.state_variance:.1f}, q={node.q_value:.1f}, var_q={node.q_variance:.1f}, uncertainty={node.uncertainty:.1f}, PUCT={node.get_puct_score(node.parent.visits, self.exploration_weight):.1f}, Obj={node.optimal_objective})"
                    )
                else:
                    print(
                        prefix
                        + f"Node-{node.level}-{index}(visits={node.visits}, prob={node.probability:.1f}, reward={node.reward:.1f}, var_reward={node.reward_variance:.1f}, value={node.state_value:.1f}, var_value={node.state_variance:.1f}, q={node.q_value:.1f}, var_q={node.q_variance:.1f}, uncertainty={node.uncertainty:.1f}, PUCT={node.get_puct_score(node.parent.visits, self.exploration_weight):.1f})"
                    )

        # Recursively print each child node at the next level with indentation
        child_prefix = (
            prefix.replace("-", " ") + "   "
        )  # Indentation for children nodes
        for i, child in enumerate(node.children):
            print(child_prefix + "|")
            self._print_node(child, child_prefix + "|--- ", i)

    def _all_nodes_fully_expanded(self, root):
        """
        Recursively checks whether all nodes in the MCTS tree
        have is_fully_expanded == True.

        Parameters
        ----------
        root : Node
            The root node of the MCTS tree.

        Returns
        -------
        bool
            True if all nodes are fully expanded, False otherwise.
        """
        if not root.is_fully_expanded():
            return False

        for child in root.children:
            if not self._all_nodes_fully_expanded(child):
                return False

        return True

    def derive_leaf_property(self):
        """
        Derive and collect properties for all leaf nodes in the MCTS tree.

        This function traverses the tree, identifies all leaf nodes, and collects
        their key properties including evaluation index, optimal objective, and state value.

        Returns
        -------
        list
            List of dictionaries, each containing leaf node properties:
            - 'index': evaluation_index of the leaf node
            - 'optimal_obj': optimal_objective value of the leaf node
            - 'state_value': state_value of the leaf node
            Returns empty list if no leaf nodes found.
        """
        # Step 1: Collect all leaf nodes
        leaf_nodes = []
        self._collect_leaf_nodes(self.root, leaf_nodes)

        if not leaf_nodes:
            return []

        # Step 2: Build summary list for each leaf node
        exe_summary_list = []
        for node in leaf_nodes:
            exe_summary = {
                "index": getattr(node, "evaluation_index", None),
                "optimal_obj": getattr(node, "optimal_objective", None),
                "state_value": getattr(node, "state_value", None),
            }
            exe_summary_list.append(exe_summary)

        return exe_summary_list

    def _collect_leaf_nodes(self, node, leaf_list):
        """
        Recursively collect all leaf nodes in the tree.

        Parameters
        ----------
        node : FormulationNode
            Current node to examine
        leaf_list : list
            List to store leaf nodes (modified in place)
        """
        if not node.children:  # This is a leaf node
            leaf_list.append(node)
        else:
            # Recursively check all children
            for child in node.children:
                self._collect_leaf_nodes(child, leaf_list)
