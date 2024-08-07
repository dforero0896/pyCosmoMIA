import numpy as np
import jax 
import jax.numpy as jnp
import sys, os
from cosmomia.mas import cic_mas_vec
from cosmomia import py_assign_particles_to_gals, subgrid_collapse
from cosmomia.correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, naive_pk_poles
from cosmomia.displacements import enhance_short_range, interpolate_field, spherical_collapse, vel_kernel
from astropy.table import Table, vstack
from pyfcfc.boxes import py_compute_cf
from pypowspec.pypowspec import compute_auto_box
from astropy.table import Table, vstack





POS_FILES = "/home/astro/dforero/projects/desi-patchy/run-ridged/BOXpos{}OM0.315OL0.685G360V2000.0_ALPTrs6.000z1.100.dat"
VEL_FILES = "/home/astro/dforero/projects/desi-patchy/run-ridged/VE{}EULz1.100.dat"        
DISP_FILES = "/home/astro/dforero/projects/desi-patchy/run-ridged/PSI{}z1.100.dat"
DELTA_FILE = "/home/astro/dforero/projects/desi-patchy/run-ridged/deltaBOXOM0.315OL0.685G360V2000.0_ALPTrs6.000z1.100.dat"
CWEB_FILE = "/home/astro/dforero/projects/desi-patchy/run-ridged/TwebDelta_OM0.315OL0.685G360V2000.0lthD0.050z1.100.dat"
REFERENCE_CAT = "/srv/astro/projects/cosmo3d/desi/SecondGenMocks/AbacusHOD/ELG/z1.100/AbacusSummit_base_c000_ph000/ELG_real_space.fits"
COUNTS_FILE = "/home/astro/dforero/cosmo3d/desi/patchy/ELG/z1.100/calibration/n_mock.dat"
GRID_SIZE = 360
BOX_SIZE = np.array([2000.] * 3, dtype = np.float32)
BOX_MIN = np.array([0.] * 3, dtype = np.float32)
BIN_SIZE = BOX_SIZE / GRID_SIZE
k_edges = np.arange(0.02, 0.4, 0.005)
s_edges = np.geomspace(1e-2, 50, 50)
def read_alpt_vector_field(filename, size):
    return np.vstack([np.fromfile(filename.format(s), np.float32, size) for s in ['x', 'y', 'z']]).T


if __name__ == '__main__':
    import proplot as pplt
    fig, ax = pplt.subplots(nrows = 1, ncols = 2, share = 0)
    
    displacement = read_alpt_vector_field(DISP_FILES, GRID_SIZE**3)
    dm_particles = read_alpt_vector_field(POS_FILES, GRID_SIZE**3)
    velocities = read_alpt_vector_field(VEL_FILES, GRID_SIZE**3)
    dm_dens = np.fromfile(DELTA_FILE, np.float32, GRID_SIZE**3)
    dm_cw_type = np.fromfile(CWEB_FILE, np.float32, GRID_SIZE**3).astype(np.uint32)
    target_ncount = np.fromfile(COUNTS_FILE, np.float32, GRID_SIZE**3).astype(np.uint32)
    
    
    
    ref_cat = Table.read(REFERENCE_CAT).to_pandas()
    ref_cat[['x','y','z']] += 3/2 * BOX_SIZE
    ref_cat[['x','y','z']] %=  BOX_SIZE
    delta_ref = jnp.zeros([GRID_SIZE] * 3)
    delta_ref = cic_mas_vec(delta_ref,
                    ref_cat['x'].values, ref_cat['y'].values, ref_cat['z'].values, jnp.broadcast_to(jnp.array([1.]), ref_cat['x'].values.shape[0]), 
                    ref_cat['x'].values.shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_ref.shape[0],
                    True)

    delta_ref /= delta_ref.mean()
    delta_ref -= 1.
    
    k_ref, pk_ref = naive_pk(delta_ref, BOX_SIZE[0],k_edges)
    ax[0].plot(k_ref, k_ref * pk_ref, label = "ref")
    
    tpcf = py_compute_cf([ref_cat[['x', 'y', 'z']].values], [np.ones(ref_cat['x'].shape[0], dtype = ref_cat['x'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1']) # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[1].semilogx(tpcf['s'], tpcf['s'] * tpcf['multipoles'][0,0,:])
    
    fig.savefig("plots/test_subgrid.png", dpi=300)
       
    
    
    cache_name = f"data/test_cache.npz"
    if not os.path.isfile(cache_name) or 1:
        result = py_assign_particles_to_gals(dm_particles, target_ncount,
                                            GRID_SIZE, BOX_SIZE, BOX_MIN,
                                            dm_cw_type, dm_dens, displacement,
                                            #velocities, 0, 0.3 * BIN_SIZE[0], False)
                                            velocities, 0, 5, False)
        np.savez(cache_name, **result)
    else:
        result = dict(np.load(cache_name))
    print((result['is_attractor'] < 1).any())
    print((result['is_attractor'].astype(bool)).any())
    
    print(result['pos'].min(axis=0), result['pos'].max(axis=0), result['pos'].mean(axis=0), result['pos'].std(axis=0))
    ax[1].axvline(BIN_SIZE[0], ls =':')
    ax[1].axvline(np.sqrt(np.sum((BIN_SIZE**2))), ls ='--')
    
    delta_cm_raw = jnp.zeros([GRID_SIZE] * 3)
    delta_cm_raw = cic_mas_vec(delta_cm_raw,
                    result['pos'][:,0], result['pos'][:,1], result['pos'][:,2], jnp.broadcast_to(jnp.array([1.]), result['pos'].shape[0]), 
                    result['pos'].shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_cm_raw.shape[0],
                    True)
    rho_cm_raw = delta_cm_raw.copy()
    delta_cm_raw /= delta_cm_raw.mean()
    delta_cm_raw -= 1.
    
    
    
    k_, pk_ = naive_pk(delta_cm_raw, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "CosmoMIA")
    
    
    
    tpcf = py_compute_cf([result['pos']], [np.ones(result['pos'].shape[0], dtype = result['pos'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1']) # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[1].semilogx(tpcf['s'], tpcf['s'] * tpcf['multipoles'][0,0,:])
    
    #pk = compute_auto_box(result['pos'][:,0], result['pos'][:,1], result['pos'][:,2], np.ones_like(result['pos'][:,0]), 
    #                  powspec_conf_file = "tests/powspec.conf",
    #                  )
    #
    #ax[1].plot(pk['k'], pk['k'] * pk['multipoles'][:,0], label = "CosmoMIA")
       
    print("Computing displacement", flush = True)
    displacement = jnp.stack(spherical_collapse(delta_cm_raw, BOX_SIZE[0], 0.), axis = -1)
    print(displacement.shape)
                              
    
    print("Enhancing field")
    param_init = jnp.array([30, 0.1])
    smooth, step_size = param_init
    enhance_short_range_3d = (jax.vmap(enhance_short_range, in_axes = (3, None, None)))
    interpolate_field_3d = jax.jit(jax.vmap(interpolate_field, in_axes = (0, None, None, None, None, None, None)))
    
    
    #psis = [enhance_short_range(displacement[:,i].reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE), BOX_SIZE[0], 1.) for i in range(3)]
    psis = enhance_short_range_3d(displacement.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), BOX_SIZE[0], smooth)
    print(psis.shape)
    
    
    #psis = [interpolate_field(psis[i], result['pos'], BOX_MIN[0], BOX_MIN[1], BOX_MIN[2], GRID_SIZE, BOX_SIZE[0])  for i in range(3)]
    new_pos = result['pos'] + step_size * interpolate_field_3d(psis, result['pos'], BOX_MIN[0], BOX_MIN[1], BOX_MIN[2], GRID_SIZE, BOX_SIZE[0]).T
    
    new_pos += BOX_SIZE
    new_pos %= BOX_SIZE
    print(psis.shape)
    
    
    delta_cm = jnp.zeros([GRID_SIZE] * 3)
    delta_cm = cic_mas_vec(delta_cm,
                    new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                    new_pos.shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_cm.shape[0],
                    True)

    delta_cm /= delta_cm.mean()
    delta_cm -= 1.
    
    
    
    k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "CosmoMIA enhanced")
    
    fig.savefig("plots/test_subgrid.png", dpi=300)
    tpcf = py_compute_cf([np.array(new_pos)], [np.ones(np.array(new_pos).shape[0], dtype = np.array(new_pos).dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1']) # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[1].semilogx(tpcf['s'], tpcf['s'] * tpcf['multipoles'][0,0,:], ls ='--')
    
    
    ax[0].legend(loc = 'top')
    ax[1].format(yscale = 'log')
    fig.savefig("plots/test_subgrid.png", dpi=300)
    
    
    
    
    
    exit()
    k = jnp.fft.fftfreq(GRID_SIZE, d=BOX_SIZE[0]/GRID_SIZE) * 2 * jnp.pi
    rfftn_batch = jax.vmap(jnp.fft.rfftn, in_axes = (0,))
    
    
    
    @jax.jit
    def disp_to_divergence(psis):
        psis_k = rfftn_batch(psis)
        psis_k = psis_k.at[0,...].set(-1j * k[:,None,None] * psis_k[0,...])
        psis_k = psis_k.at[1,...].set(-1j * k[None,:,None] * psis_k[1,...])
        psis_k = psis_k.at[2,...].set(-1j * k[None,None,:GRID_SIZE//2+1] * psis_k[2,...])
        return jnp.fft.irfftn(psis_k.sum(axis=0), psis.shape[1:])
    
    @jax.jit
    def modify_delta(params, rho):
        smooth, step_size = params
        psis = enhance_short_range_3d(displacement.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), BOX_SIZE[0], smooth)
        rho_prime = rho * (1 + step_size * disp_to_divergence(psis))
        delta_prime = rho_prime / rho_prime.mean() - 1
        return delta_prime
    @jax.jit
    def modify_pos(params, pos):
        smooth, step_size = params
        psis = enhance_short_range_3d(displacement.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3), BOX_SIZE[0], smooth)
        new_pos = pos + step_size * interpolate_field_3d(psis, pos, BOX_MIN[0], BOX_MIN[1], BOX_MIN[2], GRID_SIZE, BOX_SIZE[0]).T
        delta_cm = jnp.zeros([GRID_SIZE] * 3)
        delta_cm = cic_mas_vec(delta_cm,
                        new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
                        new_pos.shape[0], 
                        0., 0., 0.,
                        BOX_SIZE[0],
                        delta_cm.shape[0],
                        True)

        delta_cm /= delta_cm.mean()
        delta_cm -= 1.
        return delta_cm
    
    delta_cm = modify_delta(param_init, rho_cm_raw)
    k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "CosmoMIA enhanced field")
    
    
    tpcf = py_compute_cf([np.array(new_pos)], [np.ones(np.array(new_pos).shape[0], dtype = np.array(new_pos).dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1']) # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[1].semilogx(tpcf['s'], tpcf['s'] * tpcf['multipoles'][0,0,:], ls ='--')
    
    
    delta_cm = modify_pos(param_init, result['pos'])
    k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "CosmoMIA enhanced part")
    
    
    
    target_ncount = (target_ncount / target_ncount.mean() - 1).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    k_, pk_ = naive_pk(target_ncount, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "Number counts")
    
    [a.legend(loc = 'top') for a in ax]
    fig.savefig("plots/test_subgrid.png", dpi=300)
    
    exit()
    @jax.jit
    def loss_fn(params):
        delta_cm = modify_delta(params, rho_cm_raw)       
        k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
        log_loss = jnp.mean(jnp.abs(jnp.log10(pk_) - jnp.log10(pk_ref)))
        #direct_loss = jnp.mean(k_ * (jnp.abs((pk_) - (pk_ref))))
        return log_loss #+ direct_loss
    
    @jax.jit
    def loss_fn(params):
        delta_cm = modify_pos(params, result['pos'])    
        k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
        log_loss = jnp.mean(jnp.abs(jnp.log10(pk_) - jnp.log10(pk_ref)))
        #direct_loss = jnp.mean(k_ * (jnp.abs((pk_) - (pk_ref))))
        return log_loss #+ direct_loss
        
    
    grad_fn = jax.jit(jax.grad(loss_fn))
    print(grad_fn(param_init))
    
    import optax
    LEARNING_RATE=0.5e-1
    
    optim = optax.adam(LEARNING_RATE)
    
    def train(params, steps, print_every = 5):
        opt_state = optim.init(param_init)
        @jax.jit
        def make_step(param, opt_state):
            loss_value, grads = jax.value_and_grad(loss_fn)(param)
            updates, opt_state = optim.update(grads, opt_state, param)
            param = optax.apply_updates(param, updates)
            return param, opt_state, loss_value
        for step in range(steps):
            params, opt_state, train_loss = make_step(params, opt_state)
            if (step % print_every) == 0 or (step == steps - 1):
                print(
                    f"{step=}, train_loss={train_loss.item()}, "
                )
        return params
    params = train(param_init, 10, print_every = 5)
    print(f"Final params {params}")
    delta_cm = modify_delta(params, rho_cm_raw)
    k_, pk_ = naive_pk(delta_cm, BOX_SIZE[0], k_edges)
    ax[0].plot(k_, k_ * pk_, label = "CMIA field opt", ls ='--')
    
    
    [a.legend(loc = 'top') for a in ax]
    fig.savefig("plots/test_subgrid.png", dpi=300)