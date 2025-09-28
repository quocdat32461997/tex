from typing import Any, Dict, List


def update_lookup(
    tu_dict: Dict[str, List[Any]],
    tbu_dict: Dict[str, List[Any]],
) -> Dict[str, Any]:
    assert isinstance(tbu_dict, Dict) and isinstance(tu_dict, Dict)

    if isinstance(tu_dict, dict) is False:
        tu_dict = {}

    for key, val in tbu_dict.items():
        if key not in tu_dict.keys():
            tu_dict[key] = []
        tu_dict[key].append(val)
    return tu_dict


def update_list(
    tu_list: List[Any],
    tbu_list: List[Any],
) -> List[Any]:
    assert isinstance(tbu_list, List) and isinstance(tu_list, List)

    if len(tbu_list) > 0:
        tu_list.extend(tbu_list)
        return tu_list
    else:
        return []


def reset_list(
    tu_list: List[Any],
    tbu_list: List[Any],
) -> List[Any]:

    return []
