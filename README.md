# Xmap

An alternative `xmap` implementation for [Jax](https://github.com/google/jax).

Jax famously has a [`vmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html) vectorizing function, which lets you batch a function along a given axis. Making it trivial to take a function and apply it to arrays efficiently.
However, lines like `vmap(f, (0, 1), 0)` can lack readability, to the point that they are often commented along the lines of `([b,a], [a,b]) -> [b]`.

The `experimental.maps` namespace contains a [`xmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.maps.xmap.html) function that solves this problem elegantly using named axis (the prior example would become something like `xmap(f, in_axes=(['b', ...], [..., 'b']), out_axes=['b'])`) *and* lets you vectorize over multiple axis simultaneously (something that would require consecutive calls to `vmap` and quickly becomes messy).
But, `xmap` encapsulate a lot of other things (like resource repartition and jitting the code), [does not play well wit static arguments](https://github.com/google/jax/issues/10741), and overall is in the `experimental` namespace for a reason: it is the part of your codebase that is the most likely to break when you update your Jax version.

This library provide you with a `xmap` implementation that:

* Focuses on giving you a good interface to vectorize a function along one or more axes (it implements *none* of the other functionalities supported by the official implementation), 
* jits down to several calls to `vmap` (making it resilient to update on Jax's side),
* provides basic jit-time checks to try and catch common error (such as forgetting an argument or passing something of the wrong shape / type) with nice error messages.

## Installation

Copy the `xmap.py` file into your project.

## Usage

Here is a usage example taken from production code:

```python
f_batched = xmap(
    f,
    in_axes={
        'step_length': int,
        'det_data': [..., ...],
        'use_flag': bool,
        'flag_data': ['n_intervals', ...],
        'flag_mask': int,
        'amplitude_offset': int,
        'amplitude_view_offset': ['n_intervals'],
        'block_indices': ['n_intervals', 'blocks_per_interval'],
        'interval_starts': ['n_intervals'],
        'interval_ends': ['n_intervals'],
    },
    out_axes=(
        ['n_intervals', 'blocks_per_interval'],
        ['n_intervals', 'blocks_per_interval'],
    ),
)
```

Usage is similar to the [official implementation](https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.maps.xmap.html). Note that:

* Inputs / outputs are described as a single value (of a given type) or an array (described by a list of named or unnamed dimensions),
* inputs axes are described as a *dictionary* (which lets us name each input for improved readability),
* Inputs / outputs can be single values but also tuples and other pytrees.

All axes named in the inputs have to be present in the outputs. The result function (`f_batched`) is equivalent to a function that would run one loop per named axis and pass the resulting sliced data to `f`.
