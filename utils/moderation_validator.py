import os
import re
from collections.abc import Mapping

from openai import OpenAI, OpenAIError


def validate_moderation_case(moderation_case):
    """
    Validate a moderation case with three ordered checks:

    1. Ensure both "safe_sample" and "unsafe_sample" exist.
    2. Ensure both samples contain at least 200 English words.
    3. Use the official OpenAI Moderation API to verify:
       - safe_sample is not harmful.
       - unsafe_sample is harmful.

    Returns True when all validations pass.

    Raises:
        TypeError: If moderation_case is not a mapping.
        KeyError: If a required key is missing.
        ValueError: If a sample is invalid, too short, or fails moderation expectations.
        RuntimeError: If the OpenAI API key is missing or the moderation API call fails.
    """

    # Step 1: Validate required fields.
    if not isinstance(moderation_case, Mapping):
        raise TypeError(
            "Invalid moderation_case: expected a mapping object such as a dict. "
            "Reason: the validator must read both 'safe_sample' and 'unsafe_sample'. "
            "Please retry with a valid moderation_case object and call "
            "validate_moderation_case(moderation_case) again."
        )

    if "safe_sample" not in moderation_case:
        raise KeyError(
            "Invalid moderation_case: missing required field 'safe_sample'. "
            "Reason: the validator cannot verify the harmless sample without this field. "
            "Please retry by extracting a new data pair that includes 'safe_sample' "
            "and call validate_moderation_case(moderation_case) again."
        )

    if "unsafe_sample" not in moderation_case:
        raise KeyError(
            "Invalid moderation_case: missing required field 'unsafe_sample'. "
            "Reason: the validator cannot verify the harmful sample without this field. "
            "Please retry by extracting a new data pair that includes 'unsafe_sample' "
            "and call validate_moderation_case(moderation_case) again."
        )

    safe_sample = moderation_case["safe_sample"]
    unsafe_sample = moderation_case["unsafe_sample"]

    if not isinstance(safe_sample, str) or not safe_sample.strip():
        raise ValueError(
            "Invalid safe_sample: expected a non-empty string. "
            "Reason: the validator can only check text samples. "
            "Please retry with a valid safe_sample string and call "
            "validate_moderation_case(moderation_case) again."
        )

    if not isinstance(unsafe_sample, str) or not unsafe_sample.strip():
        raise ValueError(
            "Invalid unsafe_sample: expected a non-empty string. "
            "Reason: the validator can only check text samples. "
            "Please retry with a valid unsafe_sample string and call "
            "validate_moderation_case(moderation_case) again."
        )

    # Step 2: Validate minimum English word length.
    # This regex counts English words with optional internal apostrophes or hyphens.
    safe_words = re.findall(r"\b[A-Za-z]+(?:['-][A-Za-z]+)*\b", safe_sample)
    unsafe_words = re.findall(r"\b[A-Za-z]+(?:['-][A-Za-z]+)*\b", unsafe_sample)

    if len(safe_words) < 50:
        raise ValueError(
            f"Invalid safe_sample length: found {len(safe_words)} English words, "
            "but at least 50 English words are required. "
            "Reason: the safe_sample is too short to satisfy the dataset validation rule. "
            "Please retry by extracting a longer safe_sample and call "
            "validate_moderation_case(moderation_case) again."
        )

    if len(unsafe_words) < 200:
        raise ValueError(
            f"Invalid unsafe_sample length: found {len(unsafe_words)} English words, "
            "but at least 200 English words are required. "
            "Reason: the unsafe_sample is too short to satisfy the dataset validation rule. "
            "Please retry by extracting a longer unsafe_sample and call "
            "validate_moderation_case(moderation_case) again."
        )

    # Step 3: Validate harmfulness with the official OpenAI Moderation API.
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OpenAI API validation failed: environment variable OPENAI_API_KEY is not set. "
            "Reason: the official OpenAI Moderation API requires authentication. "
            "Please set OPENAI_API_KEY, retry, and call "
            "validate_moderation_case(moderation_case) again."
        )

    try:
        client = OpenAI()

        moderation_response = client.moderations.create(
            model="omni-moderation-latest",
            input=[safe_sample, unsafe_sample],
        )

    except OpenAIError as exc:
        raise RuntimeError(
            "OpenAI API validation failed while calling the Moderation API. "
            f"Reason: {exc}. "
            "Please retry after checking your API key, network connection, account access, "
            "and model availability, then call validate_moderation_case(moderation_case) again."
        ) from exc

    except Exception as exc:
        raise RuntimeError(
            "OpenAI API validation failed due to an unexpected error. "
            f"Reason: {exc}. "
            "Please retry after checking your runtime environment, then call "
            "validate_moderation_case(moderation_case) again."
        ) from exc

    if not getattr(moderation_response, "results", None) or len(moderation_response.results) != 2:
        raise RuntimeError(
            "OpenAI API validation failed: expected exactly 2 moderation results, "
            f"but received {len(getattr(moderation_response, 'results', []) or [])}. "
            "Reason: the validator sent safe_sample and unsafe_sample as a two-item input list. "
            "Please retry the API call and call validate_moderation_case(moderation_case) again."
        )

    safe_result = moderation_response.results[0]
    unsafe_result = moderation_response.results[1]

    if safe_result.flagged:
        safe_categories = safe_result.categories.model_dump()
        safe_scores = safe_result.category_scores.model_dump()

        safe_flagged_categories = [
            category
            for category, is_flagged in safe_categories.items()
            if is_flagged
        ]

        safe_top_scores = sorted(
            safe_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        raise ValueError(
            "Invalid safe_sample moderation result: safe_sample was flagged as harmful "
            "by the OpenAI Moderation API. "
            f"Flagged categories: {safe_flagged_categories}. "
            f"Top category scores: {safe_top_scores}. "
            "Reason: safe_sample is expected to be harmless, but the API classified it "
            "as potentially harmful. "
            "Please retry by extracting a new safe_sample and call "
            "validate_moderation_case(moderation_case) again."
        )

    if not unsafe_result.flagged:
        unsafe_categories = unsafe_result.categories.model_dump()
        unsafe_scores = unsafe_result.category_scores.model_dump()

        unsafe_top_scores = sorted(
            unsafe_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]

        raise ValueError(
            "Invalid unsafe_sample moderation result: unsafe_sample was not flagged as harmful "
            "by the OpenAI Moderation API. "
            f"Detected categories: {unsafe_categories}. "
            f"Top category scores: {unsafe_top_scores}. "
            "Reason: unsafe_sample is expected to be harmful, but the API classified it "
            "as not flagged. "
            "Please retry by extracting a new unsafe_sample and call "
            "validate_moderation_case(moderation_case) again."
        )

    return True