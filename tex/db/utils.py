import json
from typing import Any, Dict

from tex.db.constants import FORM_DATA_PATH, STATEMENT_DATA_PATH


def get_form_lines(
    year: int,
    form_name: str,
) -> Dict[str, Any]:

    with open(
        FORM_DATA_PATH.format(
            year=year,
            type="form_db",
            doc_type="forms",
            form_name=form_name,
        ),
        "r",
    ) as file:
        lines = json.load(file)["lines"]
    return lines


def get_instruction(year: int, instruction_name: str) -> Dict[str, Any]:
    with open(
        FORM_DATA_PATH.format(
            year=year,
            form="form_db",
            doc_type="instructions",
            instruction_name=instruction_name,
        ),
        "r",
    ) as file:
        return json.load(file)


def get_statements(year_int: int, statement_name: str) -> Dict[str, Any]:
    with open(
        STATEMENT_DATA_PATH.format(
            form_name=statement_name,
        ),
        "r",
    ) as file:
        return json.load(file)
