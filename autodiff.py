import inspect
import jax
from functools import wraps

def jaxify(func):
  import jax.numpy
  namespace = func.__globals__.copy()
  namespace['np'] = namespace['numpy'] = jax.numpy
  namespace['jaxify'] = lambda func: func
  source = inspect.getsource(func)
  exec(source, namespace)
  return wraps(func)(namespace[func.__name__])

def grad(func):
    return lambda *args,**kwargs : jax.grad(jaxify(func))(*args,**kwargs).__array__()
    