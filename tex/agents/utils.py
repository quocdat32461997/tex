from typing import Any, Dict, List


def update_lookup(
    tu_dict: Dict[str, List[Any]],
    tba_dict: Dict[str, List[Any]],
) -> Dict[str, Any]:
    assert isinstance(tba_dict, Dict) and isinstance(tu_dict, Dict)

    if isinstance(tu_dict, dict) is False:
        tu_dict = {}

    for key, val in tba_dict.items():
        if key not in tu_dict.keys():
            tu_dict[key] = []
        tu_dict[key].append(val)
    return tu_dict
