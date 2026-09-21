def find_reorder_permutation(source_shape, target_shape):
    """Finds the permutation needed to reorder target_shape to match source_shape."""
    is_same_order = source_shape == target_shape
    target_index_map = {value: index for index, value in enumerate(target_shape)}
    permutation = [target_index_map[value] for value in source_shape]
    return is_same_order, permutation
