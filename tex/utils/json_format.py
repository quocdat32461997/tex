import re


def clean_json_string(json_string: str) -> str:
    """Cleans a JSON string by removing unwanted characters and formatting issues.

    Args:
        json_string (str): The JSON string to be cleaned.

    Returns:
        str: The cleaned JSON string.
    """

    # Replace single quotes with double quotes
    cleaned = json_string.replace("'", '"')

    # Remove any markdown formatting (e.g., ```json ... ```)
    cleaned = re.sub(f"`*(json)*", "", cleaned)  # noqa
    cleaned = re.sub(f"\s+", " ", cleaned)  # noqa
    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    # Remove any trailing commas before closing braces/brackets
    cleaned = cleaned.replace(",}", "}").replace(",]", "]")
    cleaned = re.sub("^{\s+", "{", cleaned)
    cleaned = re.sub("\s+}", "}", cleaned)

    return cleaned
