"""Canonical embryo-stratified splits; independent of torch and data loading."""
def fold_split(ids, fold_num, fold_id):
    """Deterministic round-robin within each embryo -> (train_ids, val_ids)."""
    by_embryo = {}
    for s in ids:
        by_embryo.setdefault(s.split("_")[0], []).append(s)
    val = set()
    for _, ss in sorted(by_embryo.items()):
        for i, s in enumerate(sorted(ss)):
            if i % fold_num == fold_id:
                val.add(s)
    return [s for s in ids if s not in val], sorted(val)


def fold_assignments(ids, fold_num):
    if fold_num < 2 or len(ids) != len(set(ids)):
        raise ValueError("invalid fold universe")
    return {s: f for f in range(fold_num) for s in fold_split(ids, fold_num, f)[1]}

def manifest_folds(path, ids):
    import json
    from pathlib import Path
    folds=json.loads(Path(path).read_text())['folds']
    if set(folds)!=set(ids):
        raise ValueError('comparison IDs differ from the persisted full training fold universe')
    return folds
