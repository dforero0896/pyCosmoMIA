import jax
import jax.numpy as jnp





def fftk(shape, symmetric=True, finite=False, dtype=jnp.float32):
  """ Return k_vector given a shape (nc, nc, nc) and box_size
  """
  k = []
  for d in range(len(shape)):
    kd = jnp.fft.fftfreq(shape[d])
    kd *= 2 * jnp.pi
    kdshape = jnp.ones(len(shape), dtype='int')
    if symmetric and d == len(shape) - 1:
      kd = kd[:shape[d] // 2 + 1]
    kdshape = kdshape.at[d].set(len(kd))
    kd = kd.reshape(kdshape)

    k.append(kd.astype(dtype))
  del kd, kdshape
  return k


def box_muller_field(amplitude, phase, pkmesh):
    """
    Obtain Gaussian random field given uniform random numbers and Pk amplitude.
    """
    field = pkmesh**0.5 * jnp.sqrt(-jnp.where(amplitude != 1., jnp.log(amplitude), -1.)) * (jnp.cos(phase) + 1j * jnp.sin(phase))
    return jnp.fft.irfftn(field, (amplitude.shape[0],)*3, norm='ortho')

def linear_field_box_muller(mesh_shape, box_size, pk, seed, fixamp = False, inv_phase = False):
    """
    Generate initial conditions with fixed amplitude and/or inverted phase.
    """

    key, subkey1, subkey2 = jax.random.split(seed, 3)
    kvec = fftk(mesh_shape)
    kmesh = sum((kk / box_size[i] * mesh_shape[i])**2 for i, kk in enumerate(kvec))**0.5
    pkmesh = pk(kmesh) * (mesh_shape[0] * mesh_shape[1] * mesh_shape[2]) / (box_size[0] * box_size[1] * box_size[2])
    
    if fixamp:
        amplitude = jnp.ones_like(kmesh)
    else:
        amplitude = jax.random.uniform(subkey1, kmesh.shape, minval=1e-8) 
    
    phase = jax.random.uniform(subkey2, kmesh.shape, minval=1e-8) * 2 * jnp.pi
    if inv_phase:
        ret = []
        ret.append(box_muller_field(amplitude, phase, pkmesh))
        phase = (phase + jnp.pi)
        ret.append(box_muller_field(amplitude, phase, pkmesh))
        return ret
    
        
    
    return box_muller_field(amplitude, phase, pkmesh)