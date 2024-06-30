import numpy as np
import matplotlib.pyplot as plt

import scienceplots
import proplot as pplt
plt.style.use(['science', 'grid'])
pplt.rc.cycle = "bmh"
#pplt.rc.gridwidth = 1
#pplt.rc.gridcolor = 'k'
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension
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
#os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD

zfin = 0.1

@jax.jit
def run_simulation(cosmo, initial_conditions):  
    #snapshots = jnp.linspace(0.1,1.,2)
    snapshots = jnp.array([0.1, 1 / (1 + zfin), 1.])
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
STANDALONE = False
IC_SIZE = 2048
ZOOM_FAC = 8

smooth = 0.

seed = jax.random.PRNGKey(42)
h = 0.6736
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.807952, n_s=0.9649,
                      Omega_k=0., w0=-1., wa=0.)


D1 = jc.background.growth_factor(cosmo_jax, jnp.array([1. / (1 + zfin)]))




#colorbar_kw={'label': 'colorbar from lines'}
lss = [1, 0.5]
color_suff = ["", "_r"]
bs = 200
if __name__ == '__main__':
    fig_pk, ax_pk = pplt.subplots(nrows = 5, ncols = 2, sharex = True, figwidth = "18cm", hspace = 0, refaspect = 2.6, sharey = False, spanx = False)
    ax_pk.format(xlabel ='$k~[h/\mathrm{Mpc}]$', xscale = 'log')
    [a.area(np.linspace(0.01, 3), 0.95, 1.05, c = 'gray4', alpha = 1) for a in ax_pk[:,1]]
    fig_pk.savefig("plots/example_displacements_pk.png", dpi=300)
    #exit()
    #for kk, bs in enumerate([200, 800]):
    for kk, (ZOOM_FAC, IC_SIZE) in enumerate(zip([8, 1], [2048, 256])):
        fig_fields, ax_fields = pplt.subplots(nrows = 2, ncols = 3, share = 0, figwidth = "18cm", hspace = 0, wspace = 0)
        ax_fields.format(xlocator = 'null', ylocator = 'null')
        mesh_shape = [256] * 3
        box_size = [bs] * 3
        resolution = [b/m for b,m in zip(box_size, mesh_shape)]
        box_min = [0.] * 3
        k_ny = jnp.mean(jnp.pi * jnp.array(mesh_shape) / jnp.array(box_size))
        
        k = jnp.logspace(-4,1, 1000)
        k_in = 2 * jnp.pi / box_size[0]
        k_edges = jnp.linspace(k_in, k_ny, 200)
        k_edges_interp = jnp.linspace(k_in, jnp.sqrt(3) * k_ny, 200)
        pk_jax = linear_matter_power(cosmo_jax, k, a=1.)
        if STANDALONE:
            save_basename = "data/summaries_{method}_{resolution:.2e}.npz"
        else:
            if ZOOM_FAC > 1:
                save_basename = f"data/abacus_summaries_{{method}}_N2048_corner_{IC_SIZE//ZOOM_FAC}_{{resolution:.2e}}.npz"
            else:
                save_basename = f"data/abacus_summaries_{{method}}_N{IC_SIZE//ZOOM_FAC}_{{resolution:.2e}}.npz"
        
        
        #ax_pk[0].semilogx(k, pk_jax, label = '$P_L$')
        
        #cmap = "Dense"
        cmap = pplt.Colormap(('white', 'dark indigo'))
        if STANDALONE:
            pk = lambda x: jnp.exp(jnp.interp(jnp.log(x), jnp.log(k), jnp.log(pk_jax)))
            tic = time.time()
            ic_field = linear_field_box_muller(mesh_shape, box_size, pk, seed, fixamp = True, inv_phase = False)
            print(f"Generated ICSs in {time.time() - tic}s", flush=True)
            delta_lin = D1 * ic_field
        else:
            D1 = jc.background.growth_factor(cosmo_jax, jnp.array([1.]))
            box_size = [2000.] * 3
            if ZOOM_FAC > 1:
                IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N2048_corner_{IC_SIZE//ZOOM_FAC}.npy"
                box_size = [b/ZOOM_FAC for b in box_size]
            else:
                IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N{IC_SIZE//ZOOM_FAC}_down.npy"
            print(f"Loading ICs from {IC_FN}, box size", box_size, flush = True)
            ic_field = jnp.array(np.load(IC_FN))
            delta_lin = D1 * ic_field
            mesh_shape = list(delta_lin.shape)
            resolution = [b/m for b,m in zip(box_size, mesh_shape)]
            k_ny = jnp.mean(jnp.pi * jnp.array(mesh_shape) / jnp.array(box_size))
            k_in = 2 * jnp.pi / box_size[0]
            k_edges = jnp.linspace(k_in, k_ny, 200)
            k_edges_interp = jnp.linspace(k_in, jnp.sqrt(3) * k_ny, 200)
            bs = box_size[0]
        #ax_fields[1].imshow(D1 * ic_field[:20, :, :].mean(axis=0), colorbar = 'ur', vmin = None, cmap = 'coolwarm', colorbar_kw={'label': '$\delta_L$', 'length': 7, 'alpha':0.2, 'labelloc':'bottom', 'frame':False, 'tickloc':'bottom', 'locator':[-5, 0, 5]})
        #ax_fields[1].text(10, 240, bbox = True, text =  'IC')
        #fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
            
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
            np.savez(save_basename.format(method = method, resolution = resolution[0]))
            
            return delta_ev, pk_, r_, k_
            
        
        
        pos_lagrangian = jnp.linspace(0, box_size[0], mesh_shape[0])
        pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, jnp.prod(jnp.array(mesh_shape))).T
        
        
        
        if STANDALONE:
            fname = f"data/pm_result_{resolution[0]:.2e}.npy"
        else:
            if ZOOM_FAC > 1:
                fname = f"data/pm_result_{resolution[0]:.2e}_N2048_corner_{IC_SIZE//ZOOM_FAC}_z{zfin:.3f}.npy"
            else:
                fname = f"data/pm_result_{resolution[0]:.2e}_N_{IC_SIZE//ZOOM_FAC}_z{zfin:.3f}.npy"
        
        
        print(f"Attempting to retrieve {fname} from cache to avoid PM recomputation.", flush=True)
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
        
        
        #color_param = jax.nn.sigmoid(5 * (k_ - 0.1))
        color_param = 10 * (k_)
        
        #delta_ev_pm = delta_ev_pm.at[:32,:32,:32].set(jnp.nan)
        m = ax_fields[0].imshow(jnp.log10(2 + delta_ev_pm[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[0].text(10, 240, bbox = True, text =  'PM')
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
        
        #ax_pk[0].semilogx(k_, pk_pm, label = 'PM' if kk == 0 else None, c = 'k', alpha = lss[kk], zorder = 99)
        ax_pk[:,0].parametric(k_, pk_pm, color_param, cmap = f"k{color_suff[kk]}",label = 'PM' if kk == 0 else None, norm = 'log') 
        
        fig_pk.savefig(f"plots/example_displacements_pk.png", dpi=300)
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
        
        
        
        tic = time.time()
        psi = zeldovich(delta_lin, box_size[0], smooth)
        print(f"Ran Zeldovich sim in {time.time() - tic}s", flush=True)
        new_pos = jnp.zeros_like(pos_lagrangian)
        for i in range(3):
            new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
        
        delta_ev, pk_, r_, k_ = compute_summaries(new_pos, "Zeld", delta_ev_pm)
        ax_fields[1].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[1].text(10, 240, bbox = True, text =  'Zeld')
        #ax_pk[0].semilogx(k_, pk_, color = f"C{0}", alpha = lss[kk], zorder = 99)#, label = 'Zeld')    
        ax_pk[0,0].parametric(k_, pk_, color_param, cmap = f"C{0}{color_suff[kk]}", norm = 'log')#, label = 'Zeld')    
        #ax_pk[1].semilogx(k_, r_, color = f"C{0}", label = 'Zeld' if kk == 0 else None, alpha = lss[kk], zorder = 99)
        ax_pk[0,1].parametric(k_, r_, color_param, cmap = f"C{0}{color_suff[kk]}", norm = 'log') #, label = 'Zeld' if kk == 0 else None,
        if kk == 0:
            ax_pk[0,1].text(0.05, 0.15, bbox = True, text =  'Zeld', transform = 'axes')
        
        del delta_ev
        
        fig_pk.savefig(f"plots/example_displacements_pk.png", dpi=300)
        
        
        
        tic = time.time()
        psi = two_lpt_(delta_lin, box_size[0], smooth)
        print(f"Ran 2LPT sim in {time.time() - tic}s", flush=True)
        new_pos = jnp.zeros_like(pos_lagrangian)
        for i in range(3):
            new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
        
        delta_ev, pk_, r_, k_ = compute_summaries(new_pos, "2LPT", delta_ev_pm)
        ax_fields[2].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[2].text(10, 240, bbox = True, text =  '2LPT')
        #ax_pk[0].semilogx(k_, pk_, color = f"C{1}", alpha = lss[kk], zorder = 99)#, label = '2LPT')    
        #ax_pk[1].semilogx(k_, r_, color = f"C{1}", label = '2LPT' if kk == 0 else None, alpha = lss[kk], zorder = 99)  
        ax_pk[1,0].parametric(k_, pk_, color_param, cmap = f"C{1}{color_suff[kk]}", norm = 'log')
        ax_pk[1,1].parametric(k_, r_, color_param, cmap = f"C{1}{color_suff[kk]}", label = None if kk == 0 else None, norm = 'log')
        if kk == 0:
            ax_pk[1,1].text(0.05, 0.15, bbox = True, text =  '2LPT', transform = 'axes')
        del delta_ev
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)



        
        tic = time.time()
        psi = spherical_collapse(delta_lin, box_size[0], smooth)
        print(f"Ran SC sim in {time.time() - tic}s", flush=True)
        new_pos = jnp.zeros_like(pos_lagrangian)
        for i in range(3):
            new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
        
        delta_ev, pk_, r_, k_ = compute_summaries(new_pos, "SC", delta_ev_pm)
        ax_fields[3].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[3].text(10, 240, bbox = True, text =  'SC')
        #ax_pk[0].semilogx(k_, pk_, color = f"C{2}", alpha = lss[kk], zorder = 99)#, label = 'SC')
        #ax_pk[1].semilogx(k_, r_, color = f"C{2}", label = 'SC' if kk == 0 else None, alpha = lss[kk], zorder = 99)
        ax_pk[2,0].parametric(k_, pk_, color_param, cmap = f"C{2}{color_suff[kk]}", norm = 'log')
        ax_pk[2,1].parametric(k_, r_, color_param, cmap = f"C{2}{color_suff[kk]}", label = None if kk == 0 else None, norm = 'log')
        if kk == 0:
            ax_pk[2,1].text(0.05, 0.15, bbox = True, text =  'SC', transform = 'axes')
        del delta_ev
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)



        tic = time.time()
        psi = aug_lpt(delta_lin, box_size[0], smooth, 5, sc_alpha = 3)
        print(f"Ran ALPT sim in {time.time() - tic}s", flush=True)
        new_pos = jnp.zeros_like(pos_lagrangian)
        for i in range(3):
            new_pos = new_pos.at[:,i].set(pos_lagrangian[:,i] + interpolate_field(psi[i], pos_lagrangian, 0,0,0, psi[i].shape[0], box_size[i]))
        
        delta_ev, pk_, r_, k_ = compute_summaries(new_pos, "ALPT", delta_ev_pm)
        ax_fields[4].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[4].text(10, 240, bbox = True, text =  'ALPT')
        #ax_pk[0].semilogx(k_, pk_, color = f"C{3}", alpha = lss[kk], zorder = 99)#, label = 'ALPT'
        #ax_pk[1].semilogx(k_, r_, color = f"C{3}", label = 'ALPT' if kk == 0 else None, alpha = lss[kk], zorder = 99)
        ax_pk[3,0].parametric(k_, pk_, color_param, cmap = f"C{3}{color_suff[kk]}", norm = 'log')
        ax_pk[3,1].parametric(k_, r_, color_param, cmap = f"C{3}{color_suff[kk]}", label = None if kk == 0 else None, norm = 'log')
        if kk == 0:
            ax_pk[3,1].text(0.05, 0.15, bbox = True, text =  'ALPT', transform = 'axes')
        
        
        
        
        
        #for a in ax_fields[1,1:]:
        #    a.set_xlabel("$x~[\mathrm{Mpc}/h]$")
        fig_fields.colorbar(m, label = "$\log(\delta + 2)$", loc = 'bottom', pad = '0em')

        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
        
        
        
        #exit()
        tic = time.time()
        k_, transfer = compute_transfer_from_power(delta_ev, delta_ev_pm, box_size[0], k_edges_interp)
        assert (~np.isnan(transfer)).all()
        

        delta_ev = apply_transfer_func(rank_order_fields(delta_ev, apply_transfer_func(delta_ev_pm, box_size[0], transfer, k_), box_size[0],1.), box_size[0], 1 / transfer, k_)
        
        print(f"Remapped sim in {time.time() - tic}s", flush=True)
        delta_ev, pk_, r_, k_ = compute_summaries(delta_ev, "RMALPT", delta_ev_pm)
        #ax_pk[0].semilogx(k_, pk_, color = f"C{4}", alpha = lss[kk], zorder = 99)
        #ax_pk[1].semilogx(k_, r_, color = f"C{4}", label = 'RMALPT' if kk == 0 else None, alpha = lss[kk], zorder = 99)
        ax_pk[4,0].parametric(k_, pk_, color_param, cmap = f"C{4}{color_suff[kk]}", norm = 'log')
        ax_pk[4,1].parametric(k_, r_, color_param, cmap = f"C{4}{color_suff[kk]}", label = None if kk == 0 else None, norm = 'log')
        if kk == 0:
            ax_pk[4,1].text(0.05, 0.15, bbox = True, text =  'RMALPT', transform = 'axes')
        
        
        ax_fields[5].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[5].text(10, 240, bbox = True, text =  'RMALPT')
        ax_pk[:,0].format(yscale = 'log',xscale = 'log', ylabel = '$P(k)$')
        ax_pk[:,1].format(xscale = 'log', ylabel = '$C(k)$')
        
        for a in ax_fields:
            scalebar = ScaleBar(resolution[0], "Mpch", dimension = _Dimension("Mpch", rf"$\mathrm{{Mpc}}/h$"), location = "upper right", box_alpha = 0.5)
            a.add_artist(scalebar)
        
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
        fig_pk.savefig(f"plots/example_displacements_pk.png", dpi=300)
        continue
        
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
        ax_fields[8].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        ax_fields[8].text(10, 240, bbox = True, text =  'eALPT')
        
        k_, r_ = naive_rcoeff(delta_ev_pm, delta_ev, box_size[0], k_edges)
        ax_pk[1].semilogx(k_, r_, color = f"C{kk}", label = 'eALPT')
        ax_pk[1].legend(loc = 'll', ncols = 1)
        fig_fields.savefig(f"plots/example_displacements_{bs}.png", dpi=300)
        
    [a.format(ylim = (1, 1e6), yformatter = 'log', xformatter = 'log', xlim = (0.009, 3)) for a in ax_pk[:,0]]
    [a.format(ylim = (0., 1.06), xformatter = 'log', xlim = (0.009, 3)) for a in ax_pk[:,1]]
    #[a.legend(loc = 'll') for a in ax_pk[:,1]]
    
    fig_pk.savefig(f"plots/example_displacements_pk.png", dpi=300)