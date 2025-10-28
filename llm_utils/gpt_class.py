import os
import math
import re
import json
from typing import Dict, List, Optional, Tuple, Any
import requests


def compute_sentence_stats(response_json):
    """
    Compute sentence-level entropy and log-probability from OpenAI API response.

    Parameters:
    -----------
    response_json : dict
        The OpenAI API response (must contain 'logprobs' and 'top_logprobs').

    Returns:
    --------
    tuple of (float, float)
        - sentence_entropy: average entropy per token
        - sentence_logprob: average log-probability per token
    """
    try:
        token_entries = response_json["choices"][0]["logprobs"]["content"]
        if not token_entries:
            return 0.0, float("-inf")

        total_entropy = 0.0
        total_logprob = 0.0
        token_count = 0

        for entry in token_entries:
            logprob = entry.get("logprob", None)
            top_logprobs = entry.get("top_logprobs", [])

            # Accumulate logprob
            if logprob is not None:
                total_logprob += logprob

            # Compute entropy from top-k logprobs
            probs = [math.exp(lp["logprob"]) for lp in top_logprobs if "logprob" in lp]
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            total_entropy += entropy

            token_count += 1

        if token_count == 0:
            return 0.0, float("-inf")

        avg_entropy = total_entropy / token_count
        avg_logprob = total_logprob / token_count

        return avg_entropy, avg_logprob

    except (KeyError, IndexError, TypeError) as e:
        print(f"Error computing sentence stats: {e}")
        return 0.0, float("-inf")


class GPTLLM:
    """
    A wrapper class for OpenAI GPT API calls with token usage tracking and probability analysis.

    This class provides methods for making API calls to OpenAI's chat completions endpoint,
    with support for log probabilities, token usage tracking, and response parsing.

    Attributes:
        api_key (str): OpenAI API key
        model (str): Model name to use for API calls
        temperature (float): Sampling temperature for generation
        api_url (str): OpenAI API endpoint URL
        total_prompt_tokens (int): Accumulated prompt tokens across all calls
        total_completion_tokens (int): Accumulated completion tokens across all calls
        total_total_tokens (int): Total accumulated tokens across all calls
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        """
        Initialize the LLM client with optional API key and model configuration.

        Args:
            api_key: OpenAI API key (falls back to OPENAI_API_KEY environment variable)
            model: Model name to use for generation
            temperature: Sampling temperature (0.0 to 1.0)

        Raises:
            ValueError: If no API key is provided or found in environment
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.api_url = "https://api.openai.com/v1/chat/completions"

        if not self.api_key:
            raise ValueError("No OpenAI API key provided or found in environment.")

        # Token usage tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_total_tokens = 0

    def _log_token_usage(self, usage: Dict[str, int]) -> None:
        """
        Accumulate token usage from OpenAI API responses.

        Args:
            usage: Usage dictionary from OpenAI API response
        """
        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)
        self.total_total_tokens += usage.get("total_tokens", 0)

    def get_token_totals(self) -> Dict[str, int]:
        """
        Return accumulated token usage statistics.

        Returns:
            Dictionary containing total prompt, completion, and total tokens used
        """
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_total_tokens,
        }

    def _call_api_conf(
        self, prompt: str, result_keys: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float]:
        """
        Send a prompt to the OpenAI API and return parsed response with confidence score.
        Retries up to 3 times if 'Summary' keyword is not found in response.

        Args:
            prompt: The prompt to send to the API
            result_keys: Dictionary with keys as names and values as default values

        Returns:
            Tuple of (parsed_dictionary, confidence_probability)
        """
        default_values = result_keys.copy()
        keys_list = list(result_keys.keys())

        # Try up to 3 times to get a response with 'Summary'
        for attempt in range(3):
            try:
                response_content, response_json = self._make_api_request(prompt)
                if response_content is None:
                    continue

                # Check for 'Summary' keyword
                if "Summary" not in response_content:
                    print(
                        f"Attempt {attempt + 1}: 'Summary' keyword not found, retrying..."
                    )
                    continue

                # Extract confidence score only if Summary keyword exists
                confidence_prob = self._extract_key_probability(response_json)

                # Process the response with Summary keyword
                return self._process_summary_response(
                    response_content, confidence_prob, default_values, keys_list
                )

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                continue

        print("Failed to get valid response after 3 attempts")
        return default_values.copy(), 0.0

    def _make_api_request(self, prompt: str) -> Tuple[str, dict]:
        """Make API request and return response content and raw response JSON."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": 1,
        }

        try:
            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            response_json = response.json()

            # Log token usage
            if "usage" in response_json:
                self._log_token_usage(response_json["usage"])

            # Extract content first
            response_content = response_json["choices"][0]["message"]["content"].strip()
            # print(response_content)

            return response_content, response_json

        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return None, None

    def _process_summary_response(
        self,
        response_content: str,
        confidence_prob: float,
        default_values: Dict[str, Any],
        keys_list: list,
    ) -> Tuple[Dict[str, Any], float]:
        """Process response content that contains 'Summary' keyword."""
        # Extract JSON after "Summary" keyword
        print(response_content)
        json_content = self._extract_json_after_summary(response_content)

        # Parse JSON response to dictionary
        try:
            parsed_dict = json.loads(json_content)
        except json.JSONDecodeError as e:
            try:
                # Clean up the string and try again
                cleaned = json_content.strip()

                # Delete all tokens before the first "{" and after the last "}"
                first_brace = cleaned.find("{")
                last_brace = cleaned.rfind("}")
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    cleaned = cleaned[first_brace : last_brace + 1]

                # Remove extra whitespace but preserve structure
                cleaned = re.sub(r"\s+", " ", cleaned)
                parsed_dict = json.loads(cleaned)
            except json.JSONDecodeError as ne:
                print(f"Failed to parse JSON response: {e}, {ne}")
                print(f"Raw response: {response_content}")
                return default_values.copy(), None

        # Ensure the response has the expected structure
        if not isinstance(parsed_dict, dict):
            parsed_dict = default_values.copy()

        # Set defaults for missing keys
        for key in keys_list:
            parsed_dict.setdefault(key, default_values.get(key))

        return parsed_dict, confidence_prob

    def _extract_key_probability(self, response_json: Dict[str, Any]) -> float:
        """
        Extract the probability of boolean/numeric value tokens from API response.

        This function looks for "true", "false", "True", "False", "0", "1" tokens
        that appear after the "Summary" keyword in the response and returns the
        probability of the most recent occurrence.

        Args:
            response_json: Full API response JSON

        Returns:
            Probability of the value token (0.0 if not found)
        """
        try:
            logprobs_info = response_json["choices"][0]["logprobs"]["content"]

            # Handle both old and new API response formats
            if isinstance(logprobs_info, list):
                # New format: list of token objects
                tokens = [item.get("token", "") for item in logprobs_info]
                token_logprobs = [item.get("logprob", 0.0) for item in logprobs_info]
            else:
                # Old format: dictionary with tokens and token_logprobs arrays
                tokens = logprobs_info.get("tokens", [])
                token_logprobs = logprobs_info.get("token_logprobs", [])

            # Find "Summary" keyword first
            summary_idx = None
            for i, token in enumerate(tokens):
                token_clean = token.strip().lower()
                # Look for variations of "Summary" (case insensitive)
                if "summary" in token_clean:
                    summary_idx = i
                    break

            # If Summary found, search for boolean/numeric values after it
            start_search_idx = summary_idx + 1 if summary_idx is not None else 0

            # Find the first occurrence of target value token after Summary
            value_token_idx = None
            for i in range(start_search_idx, len(tokens)):
                token = tokens[i].strip().strip('"')

                # Look for boolean/numeric values (case-sensitive)
                if token in ["0", "1", "true", "false", "True", "False"]:
                    value_token_idx = i
                    break

            # Extract log probability of the found value token
            if value_token_idx is not None:
                value_token = tokens[value_token_idx].strip().strip('"')
                value_logprob = token_logprobs[value_token_idx]
                value_prob = math.exp(value_logprob)

                summary_status = (
                    f" (after Summary at index {summary_idx})"
                    if summary_idx is not None
                    else " (no Summary keyword found)"
                )
                print(
                    f"Found '{value_token}' at token index: {value_token_idx}{summary_status}, Log probability: {value_logprob:.4f}, Probability: {value_prob:.4f}"
                )

                return value_prob
            else:
                summary_msg = (
                    f" after Summary keyword (found at index {summary_idx})"
                    if summary_idx is not None
                    else ""
                )
                print(f"Could not find boolean/numeric value token{summary_msg}.")
                return 0.0

        except (KeyError, IndexError) as e:
            print(f"Error extracting confidence: {e}")
            return 0.0
        except Exception as e:
            print(f"Unexpected error in confidence extraction: {e}")
            return 0.0

    def _extract_json_after_summary(self, response_content: str) -> str:
        """
        Extract JSON content that appears after the "Summary" keyword.

        Args:
            response_content: The full response content from the API

        Returns:
            The JSON portion of the response, or the original content if "Summary" not found
        """
        # Look for "Summary" keyword (case insensitive, with optional markdown formatting)

        # Pattern to match "Summary" with optional markdown formatting (**Summary**)
        summary_pattern = r"\*\*\s*Summary\s*\*\*|Summary\s*:"
        match = re.search(summary_pattern, response_content, re.IGNORECASE)

        if match:
            # Extract everything after the "Summary" keyword
            json_start = match.end()
            json_content = response_content[json_start:].strip()
            return json_content
        else:
            # Fallback: return original content if "Summary" not found
            return response_content

    def _call_api_with_probability(self, prompt: str) -> Tuple[str, float, float]:
        """
        Call the LLM API and retrieve response with probability information.

        This method extends the base API call to include log probabilities and
        entropy calculations for uncertainty quantification.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Tuple of (response_content, sentence_entropy, sentence_log_probability)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "logprobs": True,  # Enable log probability tracking
        }

        try:
            # Make API request with timeout
            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            response_json = response.json()

            # Log token usage for monitoring
            if "usage" in response_json:
                self._log_token_usage(response_json["usage"])

            # Extract response content and probability statistics
            raw_response = response_json["choices"][0]["message"]["content"].strip()

            # Note: compute_sentence_stats function needs to be imported
            # from utils import compute_sentence_stats
            sentence_entropy, sentence_logprob = compute_sentence_stats(response_json)

            return raw_response, sentence_entropy, sentence_logprob

        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return "", 0.0, 0.0
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return "", 0.0, 0.0
        except Exception as e:
            print(f"Unexpected error: {e}")
            return "", 0.0, 0.0

    def _parse_llm_response(self, response: str) -> List[Any]:
        """
        Parse the LLM response into Python objects.

        Args:
            response: The raw LLM response string

        Returns:
            List of components parsed from the response, empty list if parsing fails
        """
        # Clean the response from markdown formatting
        clean_response = self._clean_llm_response(response)

        try:
            # Evaluate the response as Python code
            result = eval(clean_response)
            return result if isinstance(result, list) else []
        except Exception as e:
            print(f"[Error parsing LLM response]: {e}")
            print(f"[Raw response]: {response}")
            print(f"[Cleaned response]: {clean_response}")
            return []

    def _clean_llm_response(self, response: str) -> str:
        """
        Clean LLM response from markdown formatting.

        Args:
            response: The raw LLM response

        Returns:
            Cleaned response with markdown formatting removed
        """
        # Remove markdown code blocks if present
        if "```" in response:
            first_marker = response.find("```")
            if first_marker >= 0:
                # Find the closing backticks
                second_marker = response.find("```", first_marker + 3)
                if second_marker > first_marker:
                    # Extract content between markers
                    content = response[first_marker + 3 : second_marker].strip()

                    # Remove language identifier if present
                    if content.startswith("python"):
                        content = content[len("python") :].strip()

                    return content

        # Return original response if no code blocks found
        return response

    def _call_api(self, prompt: str) -> str:
        """
        Call the LLM API and retrieve response with probability information.

        This method extends the base API call to include log probabilities and
        entropy calculations for uncertainty quantification.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Tuple of (response_content, sentence_entropy, sentence_log_probability)
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        try:
            # Make API request with timeout
            response = requests.post(
                self.api_url, headers=headers, json=data, timeout=60
            )
            response.raise_for_status()
            response_json = response.json()

            # Log token usage for monitoring
            if "usage" in response_json:
                self._log_token_usage(response_json["usage"])

            # Extract response content and probability statistics
            raw_response = response_json["choices"][0]["message"]["content"].strip()
            return raw_response
        except requests.exceptions.RequestException as e:
            print(f"API request error: {e}")
            return ""
        except KeyError as e:
            print(f"Unexpected response format: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""
