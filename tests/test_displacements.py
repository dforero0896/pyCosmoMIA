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
from displacements import aug_lpt, spherical_collapse, two_lpt, zeldovich, divergence_to_displacement, interpolate_field, two_lpt_, nbody, lpt_init, gen_ode_func, eul_aug_lpt
from ics import linear_field_box_muller
from mas import cic_mas_vec
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff

from jaxpm.painting import cic_paint
from jaxpm.pm import linear_field, lpt, make_ode_fn
from jax.experimental.ode import odeint
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
mesh_shape = [256] * 3
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
    fig.savefig("plots/test_displacements.png", dpi=300)
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
    
    
    if not os.path.isfile(f"data/pm_result_{resolution[0]:.2e}.npy"):
        tic = time.time()
        result = run_simulation(cosmo_jax, ic_field)
        print(f"Ran JaxPM sim in {time.time() - tic}s", flush=True)
        new_pos = result
        np.save(f"data/pm_result_{resolution[0]:.2e}.npy", new_pos)    
        del result
    else:
        new_pos = np.load(f"data/pm_result_{resolution[0]:.2e}.npy")
    del ic_field
    
   
    delta_ev_pm = jnp.zeros(mesh_shape)
    delta_ev_pm = cic_mas_vec(delta_ev_pm,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev_pm.shape[0],
                    True)
    
    delta_ev_pm /= delta_ev_pm.mean()
    delta_ev_pm -= 1.
    ax[2].imshow(jnp.log10(2 + delta_ev_pm[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    ax[2].format(title = 'JaxPM')
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    
    
        
    fig.savefig("plots/test_displacements.png", dpi=300)
    tic = time.time()
    psi = zeldovich(delta_lin, box_size[0], smooth)
    print(f"Ran Zeldovich sim in {time.time() - tic}s", flush=True)
    new_pos = jnp.zeros_like(pos_lagrangian)
    for i in range(3):
        new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
    
    delta_ev = jnp.zeros(mesh_shape)
    delta_ev = cic_mas_vec(delta_ev,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    ax[3].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    ax[3].format(title = 'Zeld')
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    
    k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
    ax[4].semilogx(k_, r_, label = 'Zeld')
    
    del delta_ev
    
    
    #psi = divergence_to_displacement(-delta_lin + (1. /7) * (delta_lin)**2, box_size[0], smooth)
    
    #psi = list(map(lambda a, b: a+b, psi, two_lpt(delta_lin, box_size[0], smooth)))
    tic = time.time()
    psi = two_lpt_(delta_lin, box_size[0], smooth)
    print(f"Ran 2LPT sim in {time.time() - tic}s", flush=True)
    new_pos = jnp.zeros_like(pos_lagrangian)
    for i in range(3):
        new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
    
    delta_ev = jnp.zeros(mesh_shape)
    delta_ev = cic_mas_vec(delta_ev,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    ax[5].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    ax[5].format(title = '2LPT')
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
    ax[4].semilogx(k_, r_, label = '2LPT')
    
    del delta_ev
    
    tic = time.time()
    psi = spherical_collapse(delta_lin, box_size[0], smooth)
    print(f"Ran SC sim in {time.time() - tic}s", flush=True)
    new_pos = jnp.zeros_like(pos_lagrangian)
    for i in range(3):
        new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
    
    delta_ev = jnp.zeros(mesh_shape)
    delta_ev = cic_mas_vec(delta_ev,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    ax[6].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    ax[6].format(title = 'SC')
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
    ax[4].semilogx(k_, r_, label = 'SC')
    del delta_ev
    
    tic = time.time()
    psi = aug_lpt(delta_lin, box_size[0], smooth, 5)
    print(f"Ran ALPT sim in {time.time() - tic}s", flush=True)
    new_pos = jnp.zeros_like(pos_lagrangian)
    for i in range(3):
        new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
    
    delta_ev = jnp.zeros(mesh_shape)
    delta_ev = cic_mas_vec(delta_ev,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    m = ax[7].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    fig.colorbar(m, label = "$\log(\delta + 2)$", loc = 'right')
    ax[7].format(title = 'ALPT')
    
    
    k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
    ax[4].semilogx(k_, r_, label = 'ALPT')
    ax[4].legend(loc = 'll', ncols = 1)
    
    
    
    ax[0].format(xformatter = 'log', yformatter = 'log', ylabel = "$P_L(k)$")
    ax[1,0].format(xformatter = 'log', ylabel = "$C(k)$", xlabel = "$k~[h/\mathrm{Mpc}]$")
    
    ax[0,1].set_ylabel("$y~[\mathrm{Mpc}/h]$")
    ax[1,1].set_ylabel("$y~[\mathrm{Mpc}/h]$")
    ax[1,1].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    ax[1,2].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    ax[1,3].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    #for a in ax[1,1:]:
    #    a.set_xlabel("$x~[\mathrm{Mpc}/h]$")
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    
    np.save("data/alpt_result.npy", new_pos)
    exit()
    
    #steps = jnp.geomspace(1 / (1 + 9), 1., 4)
    steps = jnp.array([0.1, 1])
    print(steps)
    tic = time.time()
    new_pos = eul_aug_lpt(delta_lin, box_size, smooth, 5., steps, pos_lagrangian, box_min, cosmo_jax)
    print(f"Ran eALPT sim in {time.time() - tic}s with {len(steps)} steps.", flush=True)
    np.save(f"data/ealpt_{len(steps)}_result.npy", new_pos)
    
    delta_ev = jnp.zeros(mesh_shape)
    delta_ev = cic_mas_vec(delta_ev,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    box_size[0],
                    delta_ev.shape[0],
                    True)
    delta_ev /= delta_ev.mean()
    delta_ev -= 1.
    ax[8].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
    ax[8].format(title = 'eALPT')
    
    k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
    ax[4].semilogx(k_, r_, label = 'eALPT')
    ax[4].legend(loc = 'll', ncols = 1)
    fig.savefig("plots/test_displacements.png", dpi=300)
    
    
    