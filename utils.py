def unique_name(base, existing_names):
    if base not in existing_names:
        return base
    k = 1
    while f"{base}_{k}" in existing_names:
        k += 1
    return f"{base}_{k}"
