import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import proplot as pplt
plt.style.use('science')
pplt.rc.cycle = "bmh"
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo.power import linear_matter_power, nonlinear_matter_power
import sys, os, time
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/cosmomia/")
from displacements import aug_lpt, spherical_collapse, two_lpt, zeldovich, divergence_to_displacement, interpolate_field, two_lpt_, nbody, lpt_init, gen_ode_func, eul_aug_lpt, inverse_laplace_kernel
from ics import linear_field_box_muller
from mas import cic_mas_vec
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff

from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jax.experimental.ode import odeint
from scipy import ndimage
#
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

@jax.jit
def run_simulation(cosmo, initial_conditions):  
    snapshots = jnp.linspace(0.1,1.,2)
    # Create particles
    particles = jnp.stack(jnp.meshgrid(*[jnp.arange(s) for s in mesh_shape]),axis=-1).reshape([-1,3])
    z_init = 9
    # Initial displacement
    dx, p, f = lpt(cosmo, initial_conditions, particles, 1 / (1 + z_init))
    
    # Evolve the simulation forward
    res = odeint(make_ode_fn(mesh_shape), [particles+dx, p], snapshots, cosmo, rtol=1e-5, atol=1e-5)
    
    # Return the simulation volume at requested 
    return res[0][1]  * box_size[0] / mesh_shape[0]



os.makedirs("data", exist_ok = True)
VMIN = -0.1
VMAX = 1.5

smooth = 0.
mesh_shape = [32] * 3
box_size = [200.] * 3
resolution = [b/m for b,m in zip(box_size, mesh_shape)]
box_min = [0.] * 3
seed = jax.random.PRNGKey(42)
cosmo_jax = jc.Cosmology(Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96,
                      Omega_k=0., w0=-1., wa=0.)


D1 = jc.background.growth_factor(cosmo_jax, jnp.array([1.]))
print(D1)
fig, ax = pplt.subplots(nrows = 2, ncols = 4, share = 0, figwidth = "18cm")
pos_lagrangian = jnp.linspace(0, box_size[0], mesh_shape[0])
pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, jnp.prod(jnp.array(mesh_shape))).T





#colorbar_kw={'label': 'colorbar from lines'}
if __name__ == '__main__':
    
    k = jnp.logspace(-4,1, 1000)
    k_edges = jnp.linspace(0.01, 10, 100)
    pk_jax = linear_matter_power(cosmo_jax, k, a=1.)
    ax[0].loglog(k, pk_jax, label = '$P_L$')
    fig.savefig("plots/test_conf_kernels.png", dpi=300)
    #cmap = "Dense"
    cmap = pplt.Colormap(('white', 'dark indigo'))
    
    pk = lambda x: jnp.exp(jnp.interp(jnp.log(x), jnp.log(k), jnp.log(pk_jax)))
    tic = time.time()
    ic_field = linear_field_box_muller(mesh_shape, box_size, pk, seed, fixamp = True, inv_phase = False)
    print(f"Generated ICSs in {time.time() - tic}s", flush=True)
    ax[1].imshow(D1 * ic_field[:20, :, :].mean(axis=0), colorbar = 'ur', vmin = None, cmap = 'coolwarm', colorbar_kw={'label': '$\delta_L$', 'length': 7, 'alpha':0.2, 'labelloc':'bottom', 'frame':False, 'tickloc':'bottom', 'locator':[-5, 0, 5]})
    ax[1].format(title = 'IC')
    delta_lin = D1 * ic_field
    k_, pk_ = naive_pk(delta_lin, box_size[0], k_edges)
    ax[0].loglog(k_, pk_, label = 'IC')
    ax[0].legend(loc = 'best', ncols = 1)
    ax[0].format(grid = True)
    ax[1,0].format(grid = True)
    
    inverse_lap = inverse_laplace_kernel(delta_lin, box_size[0], 0.1)
    print(inverse_lap)
    ax[2].imshow(inverse_lap[:20,:,:].mean(axis=0), colorbar = 'right')
    fig.savefig("plots/test_conf_kernels.png", dpi=300)
    
    
    inverse_lap = ndimage.laplace(delta_lin, mode = 'wrap')
    ax[3].imshow(inverse_lap[:20,:,:].mean(axis=0), colorbar = 'right')
    fig.savefig("plots/test_conf_kernels.png", dpi=300)
    
    
    inverse_lap = ndimage.gaussian_laplace(delta_lin, 0.1, mode = 'wrap')
    ax[4].imshow(inverse_lap[:20,:,:].mean(axis=0), colorbar = 'right')
    fig.savefig("plots/test_conf_kernels.png", dpi=300)