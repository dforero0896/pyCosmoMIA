import time, os
import jax.numpy as jnp
import jax
import numpy as np
from mkl_fft import _numpy_fft as mklfft
os.environ['MKL_NUM_THREADS'] = "54"
os.environ['OMP_NUM_THREADS'] = "54"



@jax.jit
def benchmark_jax(array):
    
    fwd = jnp.fft.rfftn(array)
    
    bckwd = jax.block_until_ready(jnp.fft.irfftn(fwd, array.shape))
    return bckwd
    
def benchmark_np(array):
    fwd = np.fft.rfftn(array)
    
    bckwd = np.fft.irfftn(fwd, array.shape)
    return bckwd
    
def benchmark_mkl(array):
    fwd = mklfft.rfftn(array)
    
    bckwd = mklfft.irfftn(fwd, array.shape)
    return bckwd
    
    

if __name__ == '__main__':
    size = (512,) * 3
    array = np.random.random(size).astype(np.float32)
    
    tic = time.time()
    bckwd = benchmark_jax(array)
    print(f"JAX done in {time.time() - tic}s w compilation", flush = True)
    print(bckwd[:2,:2,:2])
    
    
    tic = time.time()
    bckwd = benchmark_jax(array)
    print(f"JAX done in {time.time() - tic}", flush = True)
    print(bckwd[:2,:2,:2])
    
    
    #tic = time.time()
    #benchmark_np(array)
    #print(f"Numpy done in {time.time() - tic}", flush = True)
    
    
    tic = time.time()
    bckwd = benchmark_mkl(array)
    print(f"MKL done in {time.time() - tic}", flush = True)
    print(bckwd[:2,:2,:2])
    