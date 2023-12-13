# Copyright (c) 2023 by Nestor Demeure.
# All rights reserved.  Use of this source code is governed by
# an apache 2.0-style license that can be found in the LICENSE file.

import jax
from jax import vmap
from jax import numpy as jnp
from types import EllipsisType
from inspect import Signature, Parameter

# ----------------------------------------------------------------------------------------
# PYTREE FUNCTIONS

def make_iterable(data):
    """
    Produce a datastructure that can be iterated
    making sure we are not going to iterate on dictionary keys
    """
    return data.values() if isinstance(data, dict) else data


def is_pytree_leaf(structure):
    """
    Determines if the given structure is a leaf of a pytree.

    Args:
        structure: The structure to check.

    Returns:
        True if the structure is a leaf, False otherwise.
    """
    if isinstance(structure, jax.Array):
        # An array is considered a leaf
        return True
    if isinstance(structure, list):
        # A list with no sublists, dictionaries, or tuples is a leaf
        return not any(isinstance(elem, (list, dict, tuple)) for elem in structure)
    # All other types are not leaves
    return False


def map_pytree_leaves(func, structure, func_single_values=lambda v: v):
    """
    Applies a function to all leaves in the pytree.

    Args:
        func: The function to apply to the leaves.
        structure: The pytree to apply the function to.
        func_single_values: The function to apply to single values (non-pytree elements).

    Returns:
        The transformed pytree.
    """
    if is_pytree_leaf(structure):
        return func(structure)
    elif isinstance(structure, dict):
        return {
            k: map_pytree_leaves(func, v, func_single_values)
            for k, v in structure.items()
        }
    elif isinstance(structure, list):
        return [map_pytree_leaves(func, v, func_single_values) for v in structure]
    elif isinstance(structure, tuple):
        return tuple(map_pytree_leaves(func, v, func_single_values) for v in structure)
    else:
        return func_single_values(structure)


def pytree_to_string(pytree):
    """
    Converts a pytree to a string representation.

    Args:
        pytree: The pytree to convert.

    Returns:
        A string representation of the pytree.
    """
    if is_pytree_leaf(pytree) and isinstance(pytree, list):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"Array[{items}]"
    elif isinstance(pytree, dict):
        items = ", ".join(f"{k}: {pytree_to_string(v)}" for k, v in pytree.items())
        return f"{{{items}}}"
    elif isinstance(pytree, list):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"[{items}]"
    elif isinstance(pytree, tuple):
        items = ", ".join(pytree_to_string(item) for item in pytree)
        return f"({items})"
    elif isinstance(pytree, str):
        return pytree  # Return string without quotes
    elif isinstance(pytree, EllipsisType):
        return "..."
    elif isinstance(pytree, type):
        return pytree.__name__
    else:
        return str(pytree)  # Fallback for other types


def check_pytree_axis(data, axis, info=""):
    """
    Checks that the data's shape is concordant with the given axis.

    Args:
        data: The data to check.
        axis: The axis to check against.
        info: Additional info to include in error messages.

    Raises:
        AssertionError: If the data's shape does not match the given axis.
    """
    # goes through the axis / data
    if is_pytree_leaf(axis):
        assert (
            len(axis) == data.ndim
        ), f"{info} shape ({data.shape}) does not match provided axis ({pytree_to_string(axis)})"
    elif isinstance(axis, dict):
        assert len(axis) == len(
            data
        ), f"{info} has {len(data)} elements which does not match axis ({pytree_to_string(axis)})"
        for d, (k, a) in zip(make_iterable(data), axis.items()):
            check_pytree_axis(d, a, f"{info} '{k}'")
    elif isinstance(axis, (list, tuple)):
        assert len(axis) == len(
            data
        ), f"{info} has {len(data)} elements which does not match axis ({pytree_to_string(axis)})"
        for i, (d, a) in enumerate(zip(make_iterable(data), axis)):
            check_pytree_axis(d, a, f"{info}[{i}]")
    elif isinstance(axis, type):
        is_single_number_tracer = isinstance(data, jnp.ndarray) and (data.size == 1)
        data_type = (
            data.dtype if is_single_number_tracer else type(data)
        )  # deals with JAX tracers being sorts of arrays
        if jnp.issubdtype(axis, jnp.integer):
            # integer types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, jnp.integer
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        elif jnp.issubdtype(axis, jnp.floating):
            # float types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, jnp.floating
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        elif jnp.issubdtype(axis, bool):
            # bool types all batched together to simplify axis writing
            assert jnp.issubdtype(
                data_type, bool
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
        else:
            # other, more general, types
            assert isinstance(
                data, axis
            ), f"{info} type ({data_type.__name__}) does not match provided axis ({pytree_to_string(axis)})"
    # we do not cover the case of other single values as they are assumed to be matching


def find_in_pytree(condition, structure):
    """
    Returns the first element in a pytree for which a condition is True.

    Args:
        condition: A function that evaluates to True or False for a given value.
        structure: The pytree to search through.

    Returns:
        The first element that meets the condition, or None if no such element is found.
    """
    if is_pytree_leaf(structure):
        # Check if the leaf contains a matching element
        for value in structure:
            if condition(value):
                return value
    elif isinstance(structure, (list, tuple, dict)):
        # Check if the container has a matching sub-container
        data = structure.values() if isinstance(structure, dict) else structure
        for element in data:
            result = find_in_pytree(condition, element)
            if result is not None:
                return result
    # No match found
    return None


def filter_pytree(condition, structure):
    """
    Filters elements in the leaves of a pytree based on a condition.

    Args:
        condition: A function that returns True for elements to keep.
        structure: The pytree to filter.

    Returns:
        A new pytree with only the elements that satisfy the condition.
    """

    def filter_leaf(leaf):
        return [v for v in leaf if condition(v)]

    return map_pytree_leaves(filter_leaf, structure)


def index_in_pytree(value, structure):
    """
    Finds the index of a value in all leaves of a pytree.

    Args:
        value: The value to search for.
        structure: The pytree to search within.

    Returns:
        The pytree structure with indices of the value in its leaves, or None where the value is not present.
    """

    def index_leaf(leaf):
        return leaf.index(value) if value in leaf else None

    # Applies the function to all leaves, mapping single non-leaf values to None
    return map_pytree_leaves(index_leaf, structure, lambda x: None)


# ----------------------------------------------------------------------------------------
# FUNCTION WRAPING

def runtime_check_axis(func, in_axes, out_axes):
    """
    Wraps a function to check its axes against a specification at runtime.
    Once jitted, this function becomes a no-op, it however helps with debugging.

    Args:
        func: The function to wrap.
        in_axes: The input axes specification.
        out_axes: The output axes specification.

    Returns:
        The wrapped function.
    """

    def wrapped_func(*args):
        check_pytree_axis(args, in_axes, info="Inputs")
        output = func(*args)
        check_pytree_axis(output, out_axes, info="Output")
        return output

    return wrapped_func


def set_documentation(func, in_axes, out_axes, reference_func=None):
    """
    Adds documentation to a function following the axis specification.

    Args:
        func: The function to document.
        in_axes: The input axes specification.
        out_axes: The output axes specification.
        reference_func: The reference function for documentation. If None, 'func' is used.

    Returns:
        The function with added documentation.
    """
    reference_func = reference_func or func
    doc = f"Original documentation of {reference_func.__name__}:\n{reference_func.__doc__}\n"
    doc += "\nArgs:\n"
    for key, val in in_axes.items():
        doc += f"    {key}: {pytree_to_string(val)}\n"
    doc += "\nReturns:\n" + "    " + pytree_to_string(out_axes)
    func.__doc__ = doc

    parameters = [
        Parameter(name, Parameter.POSITIONAL_OR_KEYWORD) for name in in_axes.keys()
    ]
    func.__signature__ = Signature(parameters)
    return func


# ----------------------------------------------------------------------------------------
# XMAP

def recursive_xmap(f, in_axes, out_axes):
    """
    Implements 'xmap' by applying 'vmap' recursively.

    Args:
        f: The function to be mapped.
        in_axes: The input axes mappings for 'f'.
        out_axes: The output axes mappings for 'f'.

    Returns:
        The transformed function.
    """
    # Finds the first named output axis
    named_output_axis = find_in_pytree(lambda a: isinstance(a, str), out_axes)

    # If no more named output axes, return the function with runtime axis check
    if named_output_axis is None:
        named_input_axis = find_in_pytree(lambda a: isinstance(a, str), in_axes)
        assert (
            named_input_axis is None
        ), f"Unused input axis: {named_input_axis}. Check output axes."
        return runtime_check_axis(f, in_axes, out_axes)

    # Remove axis from inputs and outputs
    filtered_in_axes = filter_pytree(lambda a: a != named_output_axis, in_axes)
    filtered_out_axes = filter_pytree(lambda a: a != named_output_axis, out_axes)

    # Recursively map over the remaining axes
    f_batched = recursive_xmap(f, filtered_in_axes, filtered_out_axes)

    # Get indices of the axis in inputs and outputs
    in_axes_indices = index_in_pytree(named_output_axis, list(in_axes.values()))
    out_axes_indices = index_in_pytree(named_output_axis, out_axes)

    # Apply vmap to remove the current axis
    f_result = vmap(f_batched, in_axes=in_axes_indices, out_axes=out_axes_indices)
    return runtime_check_axis(f_result, in_axes, out_axes)


def xmap(f, in_axes, out_axes):
    """
    A wrapper function for applying 'xmap' to a given function.

    Args:
        f: The function to apply 'xmap' to.
        in_axes: The input axes mappings.
        out_axes: The output axes mappings.

    Returns:
        callable: The batched and documented version of 'f'.
    """
    # Batch the function
    f_batched = recursive_xmap(f, in_axes, out_axes)

    # Add documentation
    return set_documentation(f_batched, in_axes, out_axes, reference_func=f)
