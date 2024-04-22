# import jax.numpy as jnp

# def check_and_replace_nan(tensor):
#   """
#   Checks if a Jax tensor contains NaN values and replaces them with ones of the same shape.

#   Args:
#       tensor: The Jax tensor to check.

#   Returns:
#       A new Jax tensor with NaN values replaced by ones, or the original tensor if no NaNs were found.
#   """

#   # Check if any element is NaN
#   if jnp.any(jnp.isnan(tensor)):
#     # Create a mask of NaN values
#     nan_mask = jnp.isnan(tensor)

#     # Replace NaN values with ones using masked_fill
#     return jnp.where(nan_mask, 1.0, tensor)
#   else:
#     # No NaN values found, return the original tensor
#     return tensor
