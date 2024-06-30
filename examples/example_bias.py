import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import scienceplots
import proplot as pplt
plt.style.use('science')
pplt.rc.cycle = "bmh"
from astropy.table import Table, vstack
import sys, os, time
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/cosmomia/")
from displacements import aug_lpt, spherical_collapse, two_lpt, zeldovich, divergence_to_displacement, interpolate_field, two_lpt_, lpt_init, gen_ode_func, eul_aug_lpt, rank_order_fields, apply_transfer_func
from ics import linear_field_box_muller
from mas import cic_mas_vec
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, compute_transfer_from_power, compute_transfer_from_cross_power
from bias_models import patchy_deterministic
import optax



REFERENCE_CAT = "/srv/astro/projects/cosmo3d/desi/SecondGenMocks/AbacusHOD/ELG/z1.100/AbacusSummit_base_c000_ph000/ELG_real_space.fits"
BOX_SIZE = 2000.
GRID_SIZE = 512
LEARNING_RATE = 1e-3

fig, ax = pplt.subplots(nrows = 1, ncols = 1)

if __name__ == '__main__':
    
    
    
    k_in = 2 * jnp.pi / BOX_SIZE
    k_ny = jnp.mean(jnp.pi * GRID_SIZE / BOX_SIZE)
    k_edges = jnp.linspace(k_in, k_ny, 200)
    
    ref_cat = Table.read(REFERENCE_CAT).to_pandas()
    ref_cat[['x','y','z']] += 3/2 * BOX_SIZE
    ref_cat[['x','y','z']] %=  BOX_SIZE
    delta_ref = jnp.zeros([GRID_SIZE] * 3)
    delta_ref = cic_mas_vec(delta_ref,
                    ref_cat['x'].values, ref_cat['y'].values, ref_cat['z'].values, jnp.broadcast_to(jnp.array([1.]), ref_cat['x'].values.shape[0]), 
                    ref_cat['x'].values.shape[0], 
                    0., 0., 0.,
                    BOX_SIZE,
                    delta_ref.shape[0],
                    True)
    mean_field = delta_ref.mean()
    delta_ref /= delta_ref.mean()
    delta_ref -= 1.
    k_, pk_ref = naive_pk(delta_ref, BOX_SIZE, k_edges)  
    ax[0].loglog(k_, pk_ref, label = "ELG")
    
    fig.savefig("plots/example_bias.png", dpi=300)
    
    delta_dm = np.load("data/reference/dmdens_abacus.npy")
    
    k_, pk_ = naive_pk(delta_dm, BOX_SIZE, k_edges)  
    ax[0].loglog(k_, pk_, label = "DM")
    
    delta_dm = np.load("data/multiscale/final/dmdens_abacus_alpt_exact_complete.npy")
    
    k_, pk_ = naive_pk(delta_dm, BOX_SIZE, k_edges)  
    ax[0].loglog(k_, pk_, label = "DM ALPT")
    
    fig.savefig("plots/example_bias.png", dpi=300)
    
    
    
    k_, pk_ = naive_xpk(delta_ref, delta_dm, BOX_SIZE, k_edges)  
    ax[0].loglog(k_, pk_, label = "DM x ELG")
    
    
    fig.savefig("plots/example_bias.png", dpi=300)
    
    params_init = jnp.array([2, 0.101, 0.1])
    
    rho_pred = patchy_deterministic(params_init, delta_dm, mean_field)
    
    
    def loss_fn(params):
        assert not (jnp.isnan(delta_dm)).any()
        rho_pred = patchy_deterministic(params, delta_dm, mean_field)
        #assert not (jnp.isnan(rho_pred)).any()
        rho_pred = rho_pred / rho_pred.mean() - 1
        k_, pk_ = naive_pk(rho_pred, BOX_SIZE, k_edges)
        
        loss_pk = jnp.nanmean(jnp.abs(jnp.log(pk_) - jnp.log(pk_ref)))
        return loss_pk
    
    
    optim = optax.adamw(LEARNING_RATE)
    
    def train(params, optim, steps, print_every):
        opt_state = optim.init(params)
        #@jax.jit
        def make_step(params, opt_state,):
            loss_value, grads = jax.value_and_grad(loss_fn)(params)
            print(grads)
            assert not jnp.isnan(grads).any()
            updates, opt_state = optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss_value
        for step in range(steps):
            params, opt_state, train_loss = make_step(params, opt_state)
            if (step % print_every) == 0 or (step == steps - 1):
                print(
                    f"{step=}, train_loss={train_loss.item()}"
                )
        return params
    
    
    params = train(params_init, optim, 50, 2)
    
    
    rho_pred = patchy_deterministic(params, delta_dm, mean_field)
    rho_pred = rho_pred / rho_pred.mean() - 1
    k_, pk_ = naive_pk(rho_pred, BOX_SIZE, k_edges)
        
        
    
    ax[0].loglog(k_, pk_, label = "Patchy ELG")
    
    ax[0].legend(loc = 'top')
    fig.savefig("plots/example_bias.png", dpi=300)