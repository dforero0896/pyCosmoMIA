import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import proplot as pplt
plt.style.use('science')
pplt.rc.cycle = "bmh"
import sys, os, time
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/cosmomia/")
from ics import downsample_ics, linear_field_box_muller
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, compute_transfer_from_power, compute_transfer_from_cross_power
import jax_cosmo as jc
from jax_cosmo.power import linear_matter_power, nonlinear_matter_power
seed = jax.random.PRNGKey(42)
h = 0.6736
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.811355,#0.807952, 
                         n_s=0.9649,
                      Omega_k=0., w0=-1., wa=0.)
k = jnp.logspace(-4,1, 1000)
pk_jax = 47.30480505646196**2 * linear_matter_power(cosmo_jax, k, a=0.01)
pk_jax = linear_matter_power(cosmo_jax, k, a=1.)
D_1 = jc.background.growth_factor(cosmo_jax, jnp.array([0.01, 0.5, 1.]))
print(D_1 / D_1[0])

pk = lambda x: jnp.exp(jnp.interp(jnp.log(x), jnp.log(k), jnp.log(pk_jax)))
box_size = [2000.] * 3




if __name__ == '__main__':
    ic_fn = "/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N2048.npy"
    odir = os.path.dirname(ic_fn)
    fig, ax = pplt.subplots(nrows = 1, ncols = 2, share = 0)
    ic = np.load(ic_fn)
    ic *= 77.86540103136878
    np.save("/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N2048_down.npy", ic)
    #ic *= 77.86540103136878 / 47.30480505646196
    #ic *= 0.90504324 # A weird constant bias in Abacus ICs
    k_ny = jnp.mean(jnp.pi * jnp.array(ic.shape) / jnp.array(box_size))
    k_edges = jnp.linspace(0.005, k_ny, 200)
    k_edges_interp = jnp.linspace(0.005, jnp.sqrt(3) * k_ny, 200)
    #ax[0].loglog(k, pk_jax)
    
    try:
        k_ab, pk_ab = np.loadtxt("/home/astro/dforero/cosmo3d/AbacusSummit/ics/CLASS_power", unpack = True)
        #ax[0].loglog(k_ab, pk_ab * (77.86540103136878 / 47.30480505646196)**2, ls = '--')
    except OSError:
        print("Abacus Pk not found")
    print(ic.shape)
    #ax[0].hist(ic.ravel(), histtype = "step", label = f"DIM = {ic.shape[0]}", density = True, bins = 100)
    for kernel_size in [2, 4, 8]:
        fname = f"{odir}/ic_dens_N{ic.shape[0]}_corner_{ic.shape[0]//kernel_size}.npy"
        if not os.path.isfile(fname) or 1:
            corner_ic = ic[:ic.shape[0]//kernel_size, :ic.shape[0]//kernel_size, :ic.shape[0]//kernel_size]
            np.save(fname, corner_ic)
        else:
            corner_ic = np.load(fname)
        #corner_ic *= 77.86540103136878/47.30480505646196
        
        print(corner_ic.shape)
        ax[1].hist(corner_ic.ravel(), histtype = "step", label = f"DIM = {corner_ic.shape[0]}", density = True, bins = 100)
        
        
        ic_field = linear_field_box_muller(list(corner_ic.shape), [b / kernel_size for b in box_size], pk, seed, fixamp = True, inv_phase = False)
        ax[1].hist(ic_field.ravel(), histtype = "step", label = f"DIM = {corner_ic.shape[0]}", density = True, bins = 100, ls = '--')
        
        
        #k_, pk_ = naive_pk(corner_ic, box_size[0]/kernel_size, k_edges)  
        #ax[0].loglog(k_, pk_)
        #pk_a = pk_.copy()
        #k_, pk_ = naive_pk(ic_field, box_size[0]/kernel_size, k_edges)  
        #ax[0].loglog(k_, pk_)
        #print(jnp.sqrt(pk_ / pk_a).mean())
        #print(77.86540103136878 / D_1[-1] * D_1[0])
        #print(47.30480505646196 / D_1[-2] * D_1[0])
        fig.savefig("plots/test_ic_downsampling.png", dpi=300)
        
        
    ax[1].legend(loc = 0, ncols = 1)
    fig.savefig("plots/test_ic_downsampling.png", dpi=300)
    for kernel_size in [2, 2, 2]:
        
        fname = f"{odir}/ic_dens_N{ic.shape[0]//kernel_size}_down.npy"
        if not os.path.isfile(fname) or 1:
            ic = downsample_ics(ic, kernel_size)
            np.save(fname, ic)
        else:
            ic = np.load(fname)
        ax[0].hist(ic.ravel(), histtype = "step", label = f"DIM = {ic.shape[0]}", density = True, bins = 100)
        
        
        ic_field = linear_field_box_muller(list(ic.shape), box_size, pk, seed, fixamp = True, inv_phase = False)
        ax[0].hist(ic_field.ravel(), histtype = "step", label = f"DIM = {ic.shape[0]}", density = True, bins = 100, ls = '--')
        fig.savefig("plots/test_ic_downsampling.png", dpi=300)    
    ax[0].legend(loc = 0, ncols = 1)
    fig.savefig("plots/test_ic_downsampling.png", dpi=300)
