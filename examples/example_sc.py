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
from displacements import aug_lpt, spherical_collapse, two_lpt, zeldovich, divergence_to_displacement, interpolate_field, two_lpt_, nbody, lpt_init, gen_ode_func, eul_aug_lpt, rank_order_fields, apply_transfer_func
from ics import linear_field_box_muller
from mas import cic_mas_vec
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, compute_transfer_from_power, compute_transfer_from_cross_power

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
k_ny = jnp.mean(jnp.pi * jnp.array(mesh_shape) / jnp.array(box_size))
seed = jax.random.PRNGKey(42)
h = 0.6736
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.807952, n_s=0.9649,
                      Omega_k=0., w0=-1., wa=0.)


D1 = jc.background.growth_factor(cosmo_jax, jnp.array([1.]))
fig, ax = pplt.subplots([[1,2,3,4],
                         [5,2,3,4]], share = 0, figwidth = "18cm", wspace = (None, 0,0), hspace = (None,), refaspect = 2, refnum = 1)



#colorbar_kw={'label': 'colorbar from lines'}
if __name__ == '__main__':
    
    k = jnp.logspace(-4,1, 1000)
    k_in = 2 * jnp.pi / box_size[0]
    k_edges = jnp.linspace(k_in, k_ny, 200)
    k_edges_interp = jnp.linspace(k_in, jnp.sqrt(3) * k_ny, 200)
    pk_jax = linear_matter_power(cosmo_jax, k, a=1.)
     
    
    #ax[0].semilogx(k, pk_jax, label = '$P_L$')
    fig.savefig("plots/example_sc.png", dpi=300)
    #cmap = "Dense"
    cmap = pplt.Colormap(('white', 'dark indigo'))

    pk = lambda x: jnp.exp(jnp.interp(jnp.log(x), jnp.log(k), jnp.log(pk_jax)))
    tic = time.time()
    ic_field = linear_field_box_muller(mesh_shape, box_size, pk, seed, fixamp = True, inv_phase = False)
    print(f"Generated ICSs in {time.time() - tic}s", flush=True)
    delta_lin = D1 * ic_field


        
    def compute_summaries(pos_or_delta, method, delta_ev_pm = None):
    
        if pos_or_delta.ndim == 2:
            new_pos = pos_or_delta
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
        elif pos_or_delta.ndim == 3:
            delta_ev = pos_or_delta
        else:
            raise ValueError
        k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
        if delta_ev_pm is None: delta_ev_pm = delta_ev
        k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
        
        
        return delta_ev, pk_, r_, k_
        
    
    
    pos_lagrangian = jnp.linspace(0, box_size[0], mesh_shape[0])
    pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, jnp.prod(jnp.array(mesh_shape))).T
    
    
    
    ax[0].format(grid = True)
    ax[1,0].format(grid = True)
    
    fname = f"data/pm_result_{resolution[0]:.2e}.npy"
    
    
    if not os.path.isfile(fname):
        print("Running PM", flush=True)
        tic = time.time()
        result = run_simulation(cosmo_jax, ic_field)
        print(f"Ran JaxPM sim in {time.time() - tic}s", flush=True)
        new_pos = result
        np.save(fname, new_pos)    
        del result
    else:
        new_pos = np.load(fname)
    del ic_field
    
   
    delta_ev_pm, pk_pm, r_, k_ = compute_summaries(new_pos, "PM")
    
    
    
    #ax[0].semilogx(k_, pk_pm, label = 'PM', c = 'k')
    bins = jnp.geomspace(1e-2, 200, 100)
    counts, bins, _ = ax[0].hist(2 + delta_ev_pm.ravel(), bins = bins, histtype = 'step', label = 'PM', color = 'k', density = True)
    fig.savefig("plots/example_sc.png", dpi=300)   
    for k, alpha in enumerate([1, 1.5, 3]):
    
        tic = time.time()
        psi = spherical_collapse(delta_lin, box_size[0], smooth, alpha = alpha)
        print(f"Ran SC sim in {time.time() - tic}s", flush=True)
        new_pos = jnp.zeros_like(pos_lagrangian)
        for i in range(3):
            new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
        
        delta_ev, pk_, r_, k_ = compute_summaries(new_pos, "SC", delta_ev_pm)
        m = ax[k+1].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax[k+1].format(title = rf'$\alpha = {alpha:.1f}$', xlocator = 'null', ylocator = 'null')
        #ax[0].semilogx(k_, pk_)#, label = 'SC')
        counts, bins, _ = ax[0].hist(2 + delta_ev.ravel(), bins = bins, histtype = 'step', density = True)
        ax[4].semilogx(k_, r_, label = rf'$\alpha = {alpha:.1f}$')
        del delta_ev
        fig.savefig("plots/example_sc.png", dpi=300)
    ax[4].legend(loc = 'best', ncols = 1)
    ax[0].legend(loc = 'best', ncols = 1)
    fig.savefig("plots/example_sc.png", dpi=300)

    
    
    
    #ax[0].format(xformatter = 'log', ylabel = "$P(k)$", yscale = 'log', yformatter = 'log')
    ax[0].format(xformatter = 'log', ylabel = "PDF", xscale = 'log', yscale = 'log', yformatter = 'log', xlabel = "$2 + \delta$")
    #ax[0].set_yscale("symlog", linthresh = 1e-3)
    ax[4].format(xformatter = 'log', ylabel = "$C(k)$", xlabel = "$k~[h/\mathrm{Mpc}]$")
    
    #ax[0,1].set_ylabel("$y~[\mathrm{Mpc}/h]$")
    #ax[1,1].set_ylabel("$y~[\mathrm{Mpc}/h]$")
    #ax[1,1].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    #ax[1,2].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    #ax[1,3].set_xlabel("$x~[\mathrm{Mpc}/h]$")
    #for a in ax[1,1:]:
    #    a.set_xlabel("$x~[\mathrm{Mpc}/h]$")
    fig.colorbar(m, label = "$\log(\delta + 2)$", loc = 'bottom', cols = (2,4), pad = 0)
    ax.format(abc = True, abcloc = 'ul')
    fig.savefig("plots/example_sc.png", dpi=300)
    