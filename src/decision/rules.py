"""Small pure rule helpers for the provisional policy."""
def confidence_is_measured(source: str) -> bool:
    return source.casefold() in {"measured", "calibrated", "human-verified"}

def evidence_is_sufficient(coverage: float, count: int, threshold: float = .6) -> bool:
    return count > 0 and coverage >= threshold
