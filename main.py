# -*- coding: utf-8 -*-
import time
import os
import sys
import argparse
from datetime import datetime
import jsonlines
from llm_utils import (
    LLMEvaluator,
    EfficientLLMComponentGenerator,
)
from utils.utility_functions import refine_mamo_problem
from efficient_mcts import EMCTS


def run_single_case_efficient_mcts(
    problem_id,
    problem_text,
    ground_truth,
    output_dir,
    api_key,
    model,
    exploration_weight,
    gamma,
    max_retry_time,
    iterations,
    num_components,
):
    # log settings
    timestamp = datetime.now().strftime("%m%d%H%M")
    log_file_name = f"{problem_id}_{timestamp}"
    full_log_path = os.path.join(output_dir, f"{log_file_name}.log")
    result_note = {}

    # Redirect stdout to log file
    with open(full_log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = log_file
        # print problem details
        print("=" * 50 + " PROBLEM CONTENT " + "=" * 50)
        print(f"[Problem ID]: {problem_id}")
        print(f"[Problem Text]: {problem_text}")
        print(f"[Ground Truth]: {ground_truth}")
        result_note["id"] = problem_id
        start_time = time.time()

        # create LLM API related classes
        # Create LLM component generator
        component_generator = EfficientLLMComponentGenerator(
            api_key=api_key, model=model
        )
        # LLM evaluator
        form_evaluator = LLMEvaluator(api_key=api_key, model=model)

        # Run MCTS for formulation
        mcts = EMCTS(
            problem_description=problem_text,
            llm_component_generator=component_generator,
            llm_formulation_evaluator=form_evaluator,
            exploration_weight=exploration_weight,
            gamma=gamma,
            max_retry_time=max_retry_time,
        )

        best_solution_exe_output, optimal_objective = mcts.search(
            iterations=iterations, num_components=num_components
        )

        # check the correctness of the best formulation
        if optimal_objective is None:
            print("No optimal objective found, cannot verify correctness.")
            is_executable = False
            is_correct = False
            explanation = "No optimal objective found due to the execution error."
        else:
            is_executable = True
            is_correct, explanation = form_evaluator.verify_optimality(
                best_solution_exe_output, ground_truth
            )
        all_solutions = mcts.derive_leaf_property()
        result_note["correct"] = is_correct
        result_note["explanation"] = explanation
        result_note["executable"] = is_executable
        result_note["all_solutions"] = all_solutions

        # result_note["is_clear"] = mcts.is_clear
        print("-" * 10 + " Result " + "-" * 10)
        print(f"[Correctness]: {is_correct}, Explanation: {explanation}")
        # print(f"Execution Success: {is_executable}, Is clear: {mcts.is_clear}")
        print(f"All solutions: {all_solutions}")

        print("-" * 10 + " Token Usage " + "-" * 10)
        response_generator_token_count = component_generator.get_token_totals()
        print(
            f"[LLM Component Generator Token Usage]: {response_generator_token_count['prompt_tokens']} (prompt), {response_generator_token_count['completion_tokens']} (completion), {response_generator_token_count['total_tokens']} (total)"
        )
        llm_evaluator_token_count = form_evaluator.get_token_totals()
        print(
            f"[LLM Evaluator Token Usage]: {llm_evaluator_token_count['prompt_tokens']} (prompt), {llm_evaluator_token_count['completion_tokens']} (completion), {llm_evaluator_token_count['total_tokens']} (total)"
        )
        num_total_token = (
            response_generator_token_count["total_tokens"]
            + llm_evaluator_token_count["total_tokens"]
        )
        print(f"[Total Token Usage]: {num_total_token}")

        # # GPT-4o
        if model == "gpt-4o":
            cost = round(
                (
                    response_generator_token_count["prompt_tokens"]
                    + llm_evaluator_token_count["prompt_tokens"]
                )
                / 1000000
                * 2.5
                + (
                    response_generator_token_count["completion_tokens"]
                    + llm_evaluator_token_count["completion_tokens"]
                )
                / 1000000
                * 10.0,
                4,
            )
            print(f"[Total Cost]: {cost} USD")
        elif model == "gpt-4o-mini":
            # GPT-4o-mini
            cost = round(
                (
                    response_generator_token_count["prompt_tokens"]
                    + llm_evaluator_token_count["prompt_tokens"]
                )
                / 1000000
                * 0.15
                + (
                    response_generator_token_count["completion_tokens"]
                    + llm_evaluator_token_count["completion_tokens"]
                )
                / 1000000
                * 0.6,
                4,
            )
            print(f"[Total Cost]: {cost} USD")
        total_time = time.time() - start_time
        print(f"[Overall Running Time]: {total_time}")
        print("\n")
        result_note["num_total_token"] = num_total_token
        result_note["running_time"] = total_time
        result_note["cost"] = cost

    # Reset stdout
    sys.stdout = sys.__stdout__

    # Check file existence
    if os.path.exists(full_log_path):
        print(f"[IsCorrect]: {is_correct}, [IsExecutable]:{is_executable}")
    else:
        print("[Error]: Log file was not created.")

    return is_correct, is_executable, result_note


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Command line interface for of Ua_LLM_Solver."
    )
    parser.add_argument(
        "--dataset", default="MamoComplex", type=str, help="choose dataset"
    )
    parser.add_argument(
        "--openAI_key",
        default="openai_api_key",
        type=str,
    )
    parser.add_argument("--llm_model", default="gpt-4o-mini", type=str)
    parser.add_argument("--log_name", default="version-1", type=str)
    
    problem_list = list(range(1,2))
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    is_correct_list = []
    is_executable_list = []
    explain_list = []

    # Create output directory if it doesn't exist
    output_dir = f"output/{args.dataset}/{args.llm_model}/{args.log_name}"  # pylint: disable=invalid-name
    os.makedirs(output_dir, exist_ok=True)

    print("[Dataset]: ", args.dataset)
    # read the id, ground truth, and problem text
    with jsonlines.open(f"testset/testset_{args.dataset}.jsonl") as reader:
        for item in reader:
            problem_id = item.get("id")
            if problem_id in problem_list:
                print("[Problem ID]: ", problem_id)
                ground_truth = item.get("ground_truth")
                problem_text = item.get("question")
                if "Mamo" in args.dataset:  # correct from Mamo authors
                    problem_text = refine_mamo_problem(problem_text, args.openAI_key)[
                        "description"
                    ]
                is_correct, is_executable, result_note = run_single_case_efficient_mcts(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    ground_truth=ground_truth,
                    output_dir=output_dir,
                    api_key=args.openAI_key,
                    model=args.llm_model,
                    exploration_weight=6,
                    gamma=0.9,
                    max_retry_time=3,
                    iterations=10,
                    num_components=1,
                )
                is_correct_list.append(is_correct)
                is_executable_list.append(is_executable)
                explain_list.append(result_note)
        print("[accuracy]:", 100 * sum(is_correct_list) / len(is_correct_list), "%")
        print(
            "[execution rate]:",
            100 * sum(is_executable_list) / len(is_executable_list),
            "%",
        )
        # Save the results to a JSONl file
        json_output_path = os.path.join(
            output_dir,
            f"results_{timestamp}.json",
        )
        with jsonlines.open(json_output_path, mode="w") as writer:
            writer.write_all(explain_list)
