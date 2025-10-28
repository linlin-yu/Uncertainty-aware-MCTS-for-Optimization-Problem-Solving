# -*- coding: utf-8 -*-
import time
import os
import sys
import argparse
import io
from datetime import datetime
import jsonlines
from llm_utils import LLMEvaluator, SequentialGPT
from utils.utility_functions import refine_mamo_problem
from utils.gurobi_utils import exec_gurobipy_code


def run_single_case_seq_gpt(
    problem_id,
    problem_text,
    ground_truth,
    output_dir,
    api_key,
    model,
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
        print("-" * 40 + " Problem Content " + "-" * 40)
        print(f"[Problem ID]: {problem_id}")
        print(f"[Problem Text]: {problem_text}")
        print(f"[Ground Truth]: {ground_truth}")
        result_note["id"] = problem_id
        start_time = time.time()

        # create LLM API related classes
        # Create LLM component generator
        response_generator = SequentialGPT(api_key=api_key, model=model)
        # LLM evaluator
        llm_evaluator = LLMEvaluator(api_key=api_key, model=model)

        # Initialize variables
        execution_output = None

        print("-" * 40 + " Formulation Generation " + "-" * 40)
        result = response_generator.forward(problem_text)
        if result["success"]:
            print("-" * 40 + " Mathematical Model Formulation " + "-" * 40)
            formulation = result["formulation"]
            print(formulation)
            gurobi_code = result["code"]
            print("-" * 40 + " Gurobipy Code " + "-" * 40)
            print(gurobi_code)

            output_buffer = io.StringIO()
            execution_output, is_executable = exec_gurobipy_code(
                gurobi_code, output_buffer
            )

            print("-" * 40 + " Execution Output " + "-" * 40)
            print(execution_output)

            print("-" * 40 + " Result Verification " + "-" * 40)
            if is_executable:
                is_correct, explanation = llm_evaluator.verify_optimality(
                    execution_output, ground_truth
                )
                print(
                    f"[Solution verification]: {'Correct' if is_correct else 'Incorrect'}"
                )
                print(explanation)
            else:
                is_correct = False
                explanation = "Code is not executable."
        else:
            print("LLM Generation Error")
            is_correct = False
            explanation = "Generation fails."
            is_executable = False
        result_note["correct"] = is_correct
        result_note["explanation"] = explanation
        result_note["executable"] = is_executable

        print("-" * 10 + " Token Usage " + "-" * 10)
        response_generator_token_count = response_generator.get_token_totals()
        print(
            f"[LLM Component Generator Token Usage]: {response_generator_token_count['prompt_tokens']} (prompt), {response_generator_token_count['completion_tokens']} (completion), {response_generator_token_count['total_tokens']} (total)"
        )
        llm_evaluator_token_count = llm_evaluator.get_token_totals()
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
        # # GPT4-0613
        # print(
        #     f"[Total Cost]: {round((response_generator_token_count['prompt_tokens'] + llm_evaluator_token_count['prompt_tokens']) / 1000000 * 30 + (response_generator_token_count['completion_tokens'] + llm_evaluator_token_count['completion_tokens']) / 1000000 * 60.0, 4)} USD"
        # )
        # GPT-4o-0513
        # print(
        #     f"[Total Cost]: {round((response_generator_token_count['prompt_tokens'] + llm_evaluator_token_count['prompt_tokens']) / 1000000 * 5 + (response_generator_token_count['completion_tokens'] + llm_evaluator_token_count['completion_tokens']) / 1000000 * 15.0, 4)} USD"
        # )
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
        "--dataset", default="optmath_bench", type=str, help="choose dataset"
    )
    parser.add_argument(
        "--openAI_key",
        default="openai_api_key",
        type=str,
    )
    parser.add_argument("--llm_model", default="gpt-4o-mini", type=str)
    parser.add_argument("--log_name", default="seq-ask", type=str)
    parser.add_argument("--flag_low", default=1, type=int)
    parser.add_argument("--flag_high", default=700, type=int)

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    is_correct_list = []
    is_executable_list = []
    explain_list = []

    # Create output directory if it doesn't exist
    output_dir = f"output/{args.dataset}/{args.llm_model}/{args.log_name}" # pylint: disable=invalid-name
    os.makedirs(output_dir, exist_ok=True)

    print("[Dataset]: ", args.dataset)
    # read the id, ground truth, and problem text
    with jsonlines.open(f"testset/testset_{args.dataset}.jsonl") as reader:
        for item in reader:
            problem_id = item.get("id")
            if args.flag_low <= problem_id <= args.flag_high:
                print("[Problem ID]: ", problem_id)
                ground_truth = item.get("ground_truth")
                problem_text = item.get("question")
                if "Mamo" in args.dataset:  # correct from Mamo authors
                    problem_text = refine_mamo_problem(problem_text, args.openAI_key)[
                        "description"
                    ]
                is_correct, is_executable, result_note = run_single_case_seq_gpt(
                    problem_id=problem_id,
                    problem_text=problem_text,
                    ground_truth=ground_truth,
                    output_dir=output_dir,
                    api_key=args.openAI_key,
                    model=args.llm_model,
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
