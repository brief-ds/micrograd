
from os import environ
DTYPE = environ.get('DTYPE', 'float64')
assert DTYPE in ('float16', 'float32', 'float64', 'float128')

from .engine import Value, tensordot
