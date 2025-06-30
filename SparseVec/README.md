# SparseVector

This package implements the `SparseVector` class, a Python implementation of a sparse vector. This implementation efficiently stores and manipulates sparse vectors and is intended to be used for collaborative filtering.

## Public methods

### `sum`

Calculates the sum of all elements in the vector.

**Returns:**
- `int`: The sum of all elements in the vector.

### `increment`

Increments the value of a specific key in the vector, and updates the norm.

**Arguments:**
- `key (int)`: The key whose value is to be incremented.
- `value (int, optional)`: The amount by which to increment the key's value. Defaults to 1.

**Raises:**
- `TypeError`: If the key is not an integer.

### `scalar_multiply`

Multiplies the vector by a scalar.

**Arguments:**
- `num (int)`: The scalar value to multiply the vector by.

**Returns:**
- `SparseVector`: A new `SparseVector` instance that is the result of the multiplication.

### `dot`

Static method to compute the dot product of two vectors.

**Arguments:**
- `vec1 (SparseVector)`: The first sparse vector.
- `vec2 (SparseVector)`: The second sparse vector.

**Returns:**
- `int`: The dot product of the two vectors.