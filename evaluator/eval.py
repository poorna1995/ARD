def exact_match(pred: str, gold: str) -> bool:
    return pred.strip().lower() == gold.strip().lower()

