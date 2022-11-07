""" 
If JAX is correctly installed, it will show the correct number of our GPU devices.
"""
import jax
print(jax.local_device_count())

# from jax.lib import xla_bridge
# print(xla_bridge.get_backend().platform)
