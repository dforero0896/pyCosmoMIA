import jax 
import jax.numpy as jnp
from jax.experimental import mesh_utils
import sys
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
from cosmomia import cy_read_cic_vector, cy_cic_mas
import time
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/cosmomia/")
from displacements import interpolate_field, interpolate_field_single
from mas import cic_mas_vec, cic_mas
import proplot as pplt
    
fig, ax = pplt.subplots(nrows = 1, ncols = 2)
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    dtype = np.float32
    size = (512,) * 3
    field = np.random.uniform(size = size).astype(dtype)
    box_size = np.array([10.,10.,10.]).astype(dtype)
    box_min = np.array([5.,5.,5.]).astype(dtype)
    positions = (box_size)[None,:] * np.random.uniform(size = (int(1e7), 3)) + box_min[None,:]
    positions = positions.astype(dtype)
    tic = time.time()
    results_py = interpolate_field(field, positions, box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX toox {time.time() - tic}s", flush=True)
    tic = time.time()
    results_py = interpolate_field(field, positions, box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX toox {time.time() - tic}s", flush=True)
    #print(jax.make_jaxpr(interpolate_field)(field, positions, box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0]))
    interpolate_field_vector = jax.jit(jax.vmap(interpolate_field_single, in_axes = (None, 0, 0, 0, None, None, None, None, None)))
    tic = time.time()
    results_vmap = interpolate_field_vector(field, positions[:,0], positions[:,1], positions[:,2], box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX vmap toox {time.time() - tic}s", flush=True)
    #print(jax.make_jaxpr(interpolate_field_vector)(field, positions[:,0], positions[:,1], positions[:,2], box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0]))
    tic = time.time()
    results_vmap = interpolate_field_vector(field, positions[:,0], positions[:,1], positions[:,2], box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX vmap toox {time.time() - tic}s", flush=True)
    
    results_cy = np.zeros(positions.shape[0], dtype = positions.dtype)
    tic = time.time()
    cy_read_cic_vector["float"](results_cy, field.ravel(), positions, box_size, box_min, np.array(field.shape, dtype = np.intp), 1, 32)
    print(f"C toox {time.time() - tic}s", flush=True)
    #print(results_c)
    print(results_py)
    print(results_cy)
    print(results_vmap)
    
    
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec
    from jax.sharding import NamedSharding
    from jax.experimental import mesh_utils

    P = jax.sharding.PartitionSpec
    devices = mesh_utils.create_device_mesh((8,))
    mesh = jax.sharding.Mesh(devices, ('x',))
    sharding = jax.sharding.NamedSharding(mesh, P('x',))
    
    
    positions = jax.device_put(positions, sharding)
    #jax.debug.visualize_array_sharding(positions, use_color = False)
    tic = time.time()
    results_vmap = interpolate_field(field, positions, box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX vmap toox {time.time() - tic}s", flush=True)
    tic = time.time()
    print("shardings match:", positions.sharding == results_vmap.sharding)
    print(results_vmap)
    
    exit()
    interpolate_field_vector = jax.jit(jax.pmap(interpolate_field_single, in_axes = (None, 0, 0, 0, None, None, None, None, None)))
    tic = time.time()
    results_vmap = interpolate_field_vector(field, positions[:,0], positions[:,1], positions[:,2], box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX vmap toox {time.time() - tic}s", flush=True)
    tic = time.time()
    results_vmap = interpolate_field_vector(field, positions[:,0], positions[:,1], positions[:,2], box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    print(f"JAX vmap toox {time.time() - tic}s", flush=True)
    

    
    
    exit()
    
    
    delta_ev = jnp.zeros(size)
    tic = time.time()
    delta_ev = cic_mas_vec(delta_ev,
                    positions[:,0], positions[:,1], positions[:,2], jnp.broadcast_to(jnp.array([1.]), positions.shape[0]), 
                    positions.shape[0], 
                    *box_min,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    print(f"JAX toox {time.time() - tic}s", flush=True)
    delta_ev = delta_ev.at[...].set(0)
    tic = time.time()
    delta_ev = cic_mas_vec(delta_ev,
                    positions[:,0], positions[:,1], positions[:,2], jnp.broadcast_to(jnp.array([1.]), positions.shape[0]), 
                    positions.shape[0], 
                    *box_min,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    print(f"JAX toox {time.time() - tic}s", flush=True)
    print(delta_ev.mean(), delta_ev.std())
    print(delta_ev[:4, 0, 0])
    
    ax[0].imshow(delta_ev[:50, ...].mean(axis=0))
    
    delta_ev = np.zeros(size, dtype = dtype)
    positions = np.asarray(positions)
    w = np.ones(positions.shape[0], dtype = dtype)
    tic = time.time()
    cy_cic_mas['float'](delta_ev, positions[:,0], positions[:,1], positions[:,2], w,
                        box_min, box_size, 1, 32)
    print(f"C toox {time.time() - tic}s", flush=True)
    print(delta_ev.mean(), delta_ev.std())
    
    print(delta_ev[:4, 0, 0])
    
    ax[1].imshow(delta_ev[:50, ...].mean(axis=0))
    
    
    fig.savefig("plots/test_interp.png", dpi=300)