import time
import numpy as np
import jax 
import jax.numpy as jnp
import sys, os
from cosmomia.mas import ngp_mas_vec, ngp_mas_vec
from cosmomia import py_assign_particles_to_gals, subgrid_collapse, single_collapse_step
from cosmomia.correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, naive_pk_poles, bispec
from cosmomia.displacements import enhance_short_range, interpolate_field, spherical_collapse, vel_kernel
from astropy.table import Table, vstack
from pyfcfc.boxes import py_compute_cf
from pypowspec.pypowspec import compute_auto_box
import jax_cosmo as jc
h = 0.6736
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.807952, n_s=0.9649,
                      Omega_k=0., w0=-1., wa=0.)


s_pow = 1

REDSHIFT = 1.1
POS_FILES = "/srv/astro/projects/cosmo3d/desi/patchy/run-ridged/BOXpos{}OM0.315OL0.685G360V2000.0_ALPTrs6.000z1.100.dat"
VEL_FILES = "/srv/astro/projects/cosmo3d/desi/patchy/run-ridged/VE{}EULz1.100.dat"        
DISP_FILES = "/srv/astro/projects/cosmo3d/desi/patchy/run-ridged/PSI{}z1.100.dat"
DELTA_FILE = "/srv/astro/projects/cosmo3d/desi/patchy/run-ridged/deltaBOXOM0.315OL0.685G360V2000.0_ALPTrs6.000z1.100.dat"
CWEB_FILE = "/srv/astro/projects/cosmo3d/desi/patchy/run-ridged/TwebDelta_OM0.315OL0.685G360V2000.0lthD0.050z1.100.dat"
REFERENCE_CAT = "/srv/astro/projects/cosmo3d/desi/SecondGenMocks/AbacusHOD/ELG/z1.100/AbacusSummit_base_c000_ph000/ELG_real_space.fits"
COUNTS_FILE = "/srv/astro/projects/cosmo3d/desi/patchy/ELG/z1.100/calibration/n_mock.dat"
GRID_SIZE = 360
PK_GRID_SIZE = 360
BOX_SIZE = np.array([2000.] * 3, dtype = np.float32)
BOX_MIN = np.array([0.] * 3, dtype = np.float32)
BIN_SIZE = BOX_SIZE / GRID_SIZE
k_ny = jnp.mean(jnp.pi * jnp.array(PK_GRID_SIZE) / jnp.array(BOX_SIZE))
k_in = 2 * jnp.pi / BOX_SIZE[0]
#k_edges = jnp.linspace(k_in, k_ny, 200)
k_edges = jnp.arange(k_in, k_ny, 0.005)
s_edges = np.geomspace(1e-1, 50, 50)
def read_alpt_vector_field(filename, size):
    return np.vstack([np.fromfile(filename.format(s), np.float32, size) for s in ['x', 'y', 'z']]).T


@jax.jit
def apply_rsd(redshift, z_pos, vz, cosmo):
    efunc = jnp.sqrt(jc.background.Esqr(cosmo, 1. / (1 + redshift)))
    disp = vz * (1 + redshift) / (100 * efunc)
    return z_pos + disp

def growth_rate_approx(cosmo, redshift):
    Omega = cosmo.Omega_m * jc.background.Esqr(cosmo, 1 / (1 + redshift))**-1 * (1 + redshift)**3
    return Omega**(5. / 9)
                                     

@jax.jit
def renormalize_velocities(redshift, cosmo, vel):
    jax.debug.print("Renormalizing velocities")
    cgs_Mpc = 1.
    cgs_sec = 1.
    cgs_km = 0.3240779290e-19 * cgs_Mpc
    hconst = 1
    cvel = 1 / (cgs_km / cgs_Mpc)
    H_eval = 100 * jnp.sqrt(jc.background.Esqr(cosmo, 1 / ( 1 + redshift)))
    cpvel = jc.background.growth_rate(cosmo, jnp.atleast_1d(1 / (1 + redshift))) * H_eval / (1 + redshift)
    #cpvel = growth_rate_approx(cosmo, redshift) * H_eval / (1 + redshift)
    D1 = 1
    vel *= D1 * cvel / hconst * cpvel
    vel *= cgs_km / cgs_sec * hconst
    return vel

if __name__ == '__main__':
    import proplot as pplt
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-minimizer", default = "scipy", choices = ['scipy', 'nautilus'])
    parser.add_argument('-repopulate', action = 'store_true')
    parser.add_argument('-compute_bispec', action = 'store_true')
    parser.add_argument('-compute_rs', action = 'store_true')
    parser.add_argument('-compute_full_2pcf', action = 'store_true')
    parser.add_argument('-compute_wp', action = 'store_true')
    args = parser.parse_args()
    
    
    fig, ax = pplt.subplots(nrows = 2, ncols = 2, share = 0)
    tic = time.time()
    print("Loading ALPT", flush=True)
    displacement = read_alpt_vector_field(DISP_FILES, GRID_SIZE**3)
    dm_particles = read_alpt_vector_field(POS_FILES, GRID_SIZE**3)
    velocities = renormalize_velocities(REDSHIFT, cosmo_jax, read_alpt_vector_field(VEL_FILES, GRID_SIZE**3))
    dm_dens = np.fromfile(DELTA_FILE, np.float32, GRID_SIZE**3)
    dm_cw_type = np.fromfile(CWEB_FILE, np.float32, GRID_SIZE**3).astype(np.uint32)
    target_ncount = np.fromfile(COUNTS_FILE, np.float32, GRID_SIZE**3).astype(np.uint32)
    
    print(f"Done in {time.time() - tic}s", flush=True)
    
    
    
    ref_cat = Table.read(REFERENCE_CAT).to_pandas().astype(np.double)
    ref_cat['zrsd'] = apply_rsd(REDSHIFT, ref_cat['z'].values, ref_cat['vz'].values, cosmo_jax)
    ref_cat[['x','y','z']] += 3/2 * BOX_SIZE
    ref_cat[['x','y','z']] %=  BOX_SIZE
    ref_cat['zrsd'] += 3/2 * BOX_SIZE[2]
    ref_cat['zrsd'] %=  BOX_SIZE[2]
    
    #pk = compute_auto_box(ref_cat['x'].values, ref_cat['y'].values, ref_cat['zrsd'].values, np.ones_like(ref_cat['x'].values), 
    #                  powspec_conf_file = "tests/powspec.conf",
    #                  )
    
    
    delta_ref = jnp.zeros([PK_GRID_SIZE] * 3)
    delta_ref = ngp_mas_vec(delta_ref,
                    ref_cat['x'].values, ref_cat['y'].values, ref_cat['zrsd'].values, jnp.broadcast_to(jnp.array([1.]), ref_cat['x'].values.shape[0]), 
                    ref_cat['x'].values.shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_ref.shape[0],
                    True)
    delta_ref /= delta_ref.mean()
    delta_ref -= 1.
    k_ref, pk_ref = naive_pk_poles(delta_ref, BOX_SIZE[0],k_edges, index = 1)
    
    
    #k_ref, pk_ref = pk['k'], pk['multipoles']
    
    ax[0,0].plot(k_ref, k_ref * pk_ref[:,0], label = "ref", color = 'k')
    ax[1,0].plot(k_ref, k_ref * pk_ref[:,1], label = "ref", color = 'k')
    
    #target_ncount = jnp.array(target_ncount).at[...].set(0).reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE)
    #target_ncount = ngp_mas_vec(target_ncount,
    #                ref_cat['x'].values, ref_cat['y'].values, ref_cat['z'].values, jnp.broadcast_to(jnp.array([1.]), ref_cat['x'].values.shape[0]), 
    #                ref_cat['x'].values.shape[0], 
    #                0., 0., 0.,
    #                BOX_SIZE[0],
    #                delta_ref.shape[0],
    #                True)
    #target_ncount = np.array(target_ncount).ravel()
    tpcf = py_compute_cf([ref_cat[['x', 'y', 'zrsd']].values], [np.ones(ref_cat['x'].shape[0], dtype = ref_cat['x'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1'],
                        verbose = 'F') # CF estimator (not necessary if only pair counts are required)
    tpcf_ref = tpcf
    print(tpcf['multipoles'].shape)
    ax[0,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,0,:], color = 'k')
    ax[1,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,1,:], color = 'k')
    #[a.set_yscale('symlog') for a in ax[:,1]]
    fig.savefig("plots/example_subgrid.png", dpi=300)
    
    vel_kernel_3d = jax.jit(jax.vmap(vel_kernel, in_axes = (0, None, None)))
    #velocities = np.array(vel_kernel_3d(velocities.T.reshape(3, GRID_SIZE, GRID_SIZE, GRID_SIZE), 6., BOX_SIZE[0]).reshape(3, -1).T)
    velocities = np.array(vel_kernel_3d(velocities.T.reshape(3, GRID_SIZE, GRID_SIZE, GRID_SIZE), 8., BOX_SIZE[0]).reshape(3, -1).T)
    cache_name = f"data/test_cache.npz"
    if not os.path.isfile(cache_name) or args.repopulate:
        tic = time.time()
        print("Populating catalog with particles", flush=True)
        result_init = py_assign_particles_to_gals(dm_particles, target_ncount,
                                            GRID_SIZE, BOX_SIZE, BOX_MIN,
                                            dm_cw_type, dm_dens, displacement,
                                            velocities, 
                                            0,                  # seed
                                            0.7 * BIN_SIZE[0],  #dist_parameter stdev of gauss, mean of exponential
                                            1,                   # distribution for particles around DM 1= gaussian, 2 = exp
                                            2,                   # distribution for fully random particles around cell center 1 = gauss, 2 = exp 3 = triangle (EZmock)
                                            debug = False,
                                            )
                                            #velocities, 0, 3, False)
        print(F"Done in {time.time() - tic}", flush=True)
        np.savez(cache_name, **result_init)
        
    else:
        result_init = dict(np.load(cache_name))
       
    
    ax[0,1].axvline(BIN_SIZE[0], ls =':')
    ax[0,1].axvline(np.sqrt(np.sum((BIN_SIZE**2))), ls ='--')
    
    #pk = compute_auto_box(result['pos'][:,0], result['pos'][:,1], result['pos'][:,2], np.ones_like(result['pos'][:,2]), 
    #                  powspec_conf_file = "tests/powspec_subgrid.conf",
    #                  )
    #k_, pk_ = pk['k'], pk['multipoles']
    result_rsd = apply_rsd(REDSHIFT, result_init['pos'][:,2], result_init['vel'][:,2], cosmo_jax)
    result_rsd += BOX_SIZE[2]
    result_rsd %= BOX_SIZE[2]
    delta_cm_raw = jnp.zeros([PK_GRID_SIZE] * 3)
    delta_cm_raw = ngp_mas_vec(delta_cm_raw,
                    result_init['pos'][:,0], result_init['pos'][:,1], result_rsd, jnp.broadcast_to(jnp.array([1.]), result_init['pos'].shape[0]), 
                    result_init['pos'].shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_cm_raw.shape[0],
                    True)
    rho_cm_raw = delta_cm_raw.copy()
    delta_cm_raw /= delta_cm_raw.mean()
    delta_cm_raw -= 1.
    
    
    pax = ax.panel('bottom')
    k_, pk_ = naive_pk_poles(delta_cm_raw, BOX_SIZE[0], k_edges, index = 1)
    ax[0,0].plot(k_, k_ * pk_[:,0], label = "CosmoMIA")
    pax[0,0].plot(k_, 100 * (pk_[:,0] / pk_ref[:,0] - 1))
    ax[1,0].plot(k_, k_ * pk_[:,1], label = "CosmoMIA")
    pax[1,0].plot(k_, 100 * (pk_[:,1] / pk_ref[:,1] - 1))
    
    pax.area(k_, -2.5, 2.5, color = 'gray5', zorder = 0)
    pax.format(ylim = (-5, 5))
    
    
    
    tpcf = py_compute_cf([np.c_[result_init['pos'][:,:2], result_rsd]], [np.ones(result_init['pos'].shape[0], dtype = result_init['pos'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1'],
                        verbose = 'F') # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[0,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,0,:])
    ax[1,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,1,:])
    fig.savefig("plots/example_subgrid.png", dpi=300)
    
    ################################################################################################################33333
    # Julia pars [6.681508955504605, 5.47808573538, 0.7534895556587489, 1.1209400930989315, 0.10218062957157581]
    #params = np.array([0.75, 6.68, 1.12, 5.47, 0.1], dtype = np.float32)
    #params = np.array([0.3, 2., 0.5, 6.4, 0.1], dtype = np.float32)
    params = np.array([0.15, 3, 0.67, 6.6, 3.], dtype = np.float32)
    #########################################################################################################3
    
    
    
    tic = time.time()
    print(result_init['vel'].min(axis = 0), result_init['vel'].mean(axis = 0), result_init['vel'].max(axis = 0))
    result = subgrid_collapse(result_init, params, BOX_SIZE, result_init['is_attractor'].astype(bool), 99, 32, debug = False)
    print(f"Collapse in {time.time() - tic}s", flush = True)
    print(result['vel'].min(axis = 0), result['vel'].mean(axis = 0), result['vel'].max(axis = 0))
    
    result_rsd = apply_rsd(REDSHIFT, result['pos'][:,2], result['vel'][:,2], cosmo_jax)
    result_rsd += BOX_SIZE[2]
    result_rsd %= BOX_SIZE[2]
    
    delta_cm_raw = jnp.zeros([PK_GRID_SIZE] * 3)
    delta_cm_raw = ngp_mas_vec(delta_cm_raw,
                    result['pos'][:,0], result['pos'][:,1], result_rsd, jnp.broadcast_to(jnp.array([1.]), result['pos'].shape[0]), 
                    result['pos'].shape[0], 
                    0., 0., 0.,
                    BOX_SIZE[0],
                    delta_cm_raw.shape[0],
                    True)
    rho_cm_raw = delta_cm_raw.copy()
    delta_cm_raw /= delta_cm_raw.mean()
    delta_cm_raw -= 1.
    
    
    
    k_, pk_ = naive_pk_poles(delta_cm_raw, BOX_SIZE[0], k_edges, index = 1)
    ax[0,0].plot(k_, k_ * pk_[:,0], label = "CosmoMIA+Coll")
    pax[0,0].plot(k_, 100 * (pk_[:,0] / pk_ref[:,0] - 1))
    ax[1,0].plot(k_, k_ * pk_[:,1], label = "CosmoMIA+Coll")
    pax[1,0].plot(k_, 100 * (pk_[:,1] / pk_ref[:,1] - 1))
    
    
    
    tpcf = py_compute_cf([np.c_[result['pos'][:,:2], result_rsd]], [np.ones(result['pos'].shape[0], dtype = result['pos'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1'],
                        verbose = 'F') # CF estimator (not necessary if only pair counts are required)

    print(tpcf['multipoles'].shape)
    ax[0,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,0,:])
    ax[1,1].semilogx(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,1,:])
    ax[0,0].legend(loc = 'top')
    [a.format(xlabel = '$k~[h/\mathrm{Mpc}]$', ylabel = f"$kP_{2*i}(k)$") for i, a in enumerate(ax[:,0])]
    [a.format(xlabel = '$s~[\mathrm{Mpc}/h]$', ylabel = rf"$s^{{{s_pow}}}\xi_{2*i}(s)$") for i, a in enumerate(ax[:,1])]
    fig.savefig("plots/example_subgrid.png", dpi=300)
    
    if args.compute_bispec:
        fig, ax = pplt.subplots([[1]], share = 0)
        pax = ax.panel('bottom')
        k1, k2 = 0.1, 0.2
        theta = jnp.linspace(0, jnp.pi, 25)
        _, _, _, Bt_ref, Qt_ref = bispec(delta_ref, BOX_SIZE[0], k1, k2, theta)
        
        _, _, _, Bt, Qt = bispec(delta_cm_raw, BOX_SIZE[0], k1, k2, theta)
        
        ax[0].plot(theta, Qt_ref, c = 'k', label = 'ref')
        ax[0].plot(theta, Qt, label = 'CosmoMIA+collapse')
        pax[0].plot(theta, 100 * (Qt / Qt_ref - 1))
        pax[0].area(theta, -15, 15, color = 'gray5', zorder = 0)
        pax[0].format(ylim = (-50, 50))
        ax[0].format(xlabel = r'$\theta [\mathrm{rad}]$', ylabel = r"$Q(\theta)$")
        ax[0].legend(loc = 'top')
        
        fig.savefig("plots/example_subgrid_3pt.png", dpi=300)
    
    
    if args.compute_rs:
        
        delta_ref = jnp.zeros([PK_GRID_SIZE] * 3)
        delta_ref = ngp_mas_vec(delta_ref,
                        ref_cat['x'].values, ref_cat['y'].values, ref_cat['z'].values, jnp.broadcast_to(jnp.array([1.]), ref_cat['x'].values.shape[0]), 
                        ref_cat['x'].values.shape[0], 
                        0., 0., 0.,
                        BOX_SIZE[0],
                        delta_ref.shape[0],
                        True)
        delta_ref /= delta_ref.mean()
        delta_ref -= 1.
        k_ref, pk_ref = naive_pk_poles(delta_ref, BOX_SIZE[0],k_edges, index = 1)
        
        ax[0,0].plot(k_ref, k_ref * pk_ref[:,0], label = "ref", color = 'k')
        ax[1,0].plot(k_ref, k_ref * pk_ref[:,1], label = "ref", color = 'k')
        
        delta_cm_raw = jnp.zeros([PK_GRID_SIZE] * 3)
        delta_cm_raw = ngp_mas_vec(delta_cm_raw,
                        result['pos'][:,0], result['pos'][:,1], result['pos'][:,2], jnp.broadcast_to(jnp.array([1.]), result['pos'].shape[0]), 
                        result['pos'].shape[0], 
                        0., 0., 0.,
                        BOX_SIZE[0],
                        delta_cm_raw.shape[0],
                        True)
        rho_cm_raw = delta_cm_raw.copy()
        delta_cm_raw /= delta_cm_raw.mean()
        delta_cm_raw -= 1.
        
        
        
        k_, pk_ = naive_pk_poles(delta_cm_raw, BOX_SIZE[0], k_edges, index = 1)
        ax[0,0].plot(k_, k_ * pk_[:,0], label = "CosmoMIA+Coll", color = "C1", ls = '--')
        pax[0,0].plot(k_, 100 * (pk_[:,0] / pk_ref[:,0] - 1), color = "C1", ls = '--')
        ax[1,0].plot(k_, k_ * pk_[:,1], label = "CosmoMIA+Coll", color = "C1", ls = '--')
        pax[1,0].plot(k_, 100 * (pk_[:,1] / pk_ref[:,1] - 1), color = "C1", ls = '--')
        
        
        #k_ref, pk_ref = pk['k'], pk['multipoles']
        
        
        target_ncount = target_ncount / target_ncount.mean()
        target_ncount -= 1
        k_, pk_ = naive_pk_poles(target_ncount.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE), BOX_SIZE[0], k_edges, index = 1)
        ax[0,0].plot(k_, k_ * pk_[:,0], label = "Ncount", color = "C2", ls = '--')
        pax[0,0].plot(k_, 100 * (pk_[:,0] / pk_ref[:,0] - 1), color = "C2", ls = '--')
        ax[1,0].plot(k_, k_ * pk_[:,1], label = "Ncount", color = "C2", ls = '--')
        pax[1,0].plot(k_, 100 * (pk_[:,1] / pk_ref[:,1] - 1), color = "C2", ls = '--')
        fig.savefig("plots/example_subgrid.png", dpi=300)
        
    if args.compute_full_2pcf:
        fig, ax = pplt.subplots([[1]], share = 0)
        pax = ax.panel('bottom')
        s_edges = np.arange(0, 200, 1, dtype = ref_cat['x'].dtype)
        s_pow = 2
        tpcf = py_compute_cf([ref_cat[['x', 'y', 'zrsd']].values], [np.ones(ref_cat['x'].shape[0], dtype = ref_cat['x'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1'],
                        verbose = 'F') # CF estimator (not necessary if only pair counts are required)
        tpcf_ref = tpcf
        print(tpcf['multipoles'].shape)
        for i in range(3):
            ax[0].plot(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,i,:], color = 'k', label = 'ref' if i == 0 else None)
        
        np.save("data/full_tpcf_ref.pkl.npy", tpcf)
        fig.savefig("plots/example_subgrid_2pcf.png", dpi=300)
        
        
        tpcf = py_compute_cf([np.c_[result['pos'][:,:2], result_rsd]], [np.ones(result['pos'].shape[0], dtype = result['pos'].dtype)], 
                        s_edges.copy(), 
                        None, 
                        100, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=1, # bin type for multipoles
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        multipole = [0, 2, 4], # Multipoles to compute
                        cf = ['AA / @@ - 1'],
                        verbose = 'F') # CF estimator (not necessary if only pair counts are required)

        print(tpcf['multipoles'].shape)
        
        np.save("data/full_tpcf_coll.pkl.npy", tpcf)
        for i in range(3):
            ax[0].plot(tpcf['s'], tpcf['s']**s_pow * tpcf['multipoles'][0,i,:], label = 'CosmoMIA' if i == 0 else None)
            pax[0].plot(tpcf['s'], 100 * (tpcf['multipoles'][0,i,:] / tpcf_ref['multipoles'][0,i,:] - 1))
        ax[0].legend(loc = 'top')
        
        pax[0].area(tpcf['s'], -2.5, 2.5, color = 'gray5', zorder = 0)
        pax[0].format(ylim = (-5, 5))
        ax[0].format(xlabel = '$s~[\mathrm{Mpc}/h]$', ylabel = rf"$s^{{{s_pow}}}\xi_{2*i}(s)$")
        ax[0].legend(loc = 'top')
    
        fig.savefig("plots/example_subgrid_2pcf.png", dpi=300)
    
    
    if args.compute_wp:
        fig, ax = pplt.subplots([[1]], share = 0)
        pax = ax.panel('bottom')
        s_edges = np.geomspace(0.1, 50, 100, dtype = ref_cat['x'].dtype)
        s_pow = 1
        tpcf = py_compute_cf([ref_cat[['x', 'y', 'zrsd']].values], [np.ones(ref_cat['x'].shape[0], dtype = ref_cat['x'].dtype)], 
                        s_edges.copy(), 
                        s_edges.copy(), 
                        0, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=2, # bin type for wp
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0],    
                        wp = "T",
                        cf = ['AA / @@ - 1'],
                        verbose = 'T') # CF estimator (not necessary if only pair counts are required)
        tpcf['projected'] = np.array(tpcf['projected'])
        tpcf_ref = tpcf
        
        
        ax[0].plot(tpcf['s'], tpcf['s']**s_pow * tpcf['projected'][0,:], color = 'k', label = 'ref' )
        
        np.save("data/full_rppi_ref.pkl.npy", tpcf)
        fig.savefig("plots/example_subgrid_wp.png", dpi=300)
        
        
        tpcf = py_compute_cf([np.c_[result['pos'][:,:2], result_rsd]], [np.ones(result['pos'].shape[0], dtype = result['pos'].dtype)], 
                        s_edges.copy(), 
                        s_edges.copy(), 
                        0, 
                        label = ['A'], # Catalog labels matching the number of catalogs provided
                        bin=2, # bin type for wp
                        pair = ['AA'], # Desired pair counts
                        box=BOX_SIZE[0], 
                        wp = "T",
                        cf = ['AA / @@ - 1'],
                        verbose = 'T') # CF estimator (not necessary if only pair counts are required)
        tpcf['projected'] = np.array(tpcf['projected'])
        
        
        np.save("data/full_rppi_coll.pkl.npy", tpcf)
        
        ax[0].plot(tpcf['s'], tpcf['s']**s_pow * tpcf['projected'][0,:], label = 'CosmoMIA')
        pax[0].plot(tpcf['s'], 100 * (tpcf['projected'][0,:] / tpcf_ref['projected'][0,:] - 1))
        ax[0].legend(loc = 'top')
        
        pax[0].area(tpcf['s'], -2.5, 2.5, color = 'gray5', zorder = 0)
        pax[0].format(ylim = (-5, 5))
        ax[0].format(xlabel = '$r_p~[\mathrm{Mpc}/h]$', ylabel = rf"$s^{{{s_pow}}}w_p(s)$", xscale = "log")
        ax[0].legend(loc = 'top')
    
        fig.savefig("plots/example_subgrid_wp.png", dpi=300)
    
   
    
    
    
    
    exit()
    
    def loss_fn(params, fit_xi = False, fit_pk = True):
        
        result = subgrid_collapse(result_init, params.astype(np.float32), BOX_SIZE, result_init['is_attractor'].astype(bool), 99, debug = False)
        result_rsd = apply_rsd(REDSHIFT, result['pos'][:,2], result['vel'][:,2], cosmo_jax)
        result_rsd += BOX_SIZE[2]
        result_rsd %= BOX_SIZE[2]
        loss_val = 0
        if fit_xi:
            tpcf = py_compute_cf([np.c_[result['pos'][:,:2], result_rsd]], [np.ones(result['pos'].shape[0], dtype = result['pos'].dtype)], 
                            s_edges.copy(), 
                            None, 
                            100, 
                            label = ['A'], # Catalog labels matching the number of catalogs provided
                            bin=1, # bin type for multipoles
                            pair = ['AA'], # Desired pair counts
                            box=BOX_SIZE[0], 
                            multipole = [0, 2, 4], # Multipoles to compute
                            cf = ['AA / @@ - 1'],
                            verbose = 'F') # CF estimator (not necessary if only pair counts are required)
            loss_val += np.mean(np.abs(tpcf['s'] * (tpcf['multipoles'][0,0,:] - tpcf_ref['multipoles'][0,0,:])))
        if fit_pk:
            delta_cm_raw = jnp.zeros([GRID_SIZE] * 3)
            delta_cm_raw = ngp_mas_vec(delta_cm_raw,
                            result_init['pos'][:,0], result_init['pos'][:,1], result_rsd, jnp.broadcast_to(jnp.array([1.]), result_init['pos'].shape[0]), 
                            result_init['pos'].shape[0], 
                            0., 0., 0.,
                            BOX_SIZE[0],
                            delta_cm_raw.shape[0],
                            True)
            rho_cm_raw = delta_cm_raw.copy()
            delta_cm_raw /= delta_cm_raw.mean()
            delta_cm_raw -= 1.
            k_, pk_ = naive_pk_poles(delta_cm_raw, BOX_SIZE[0], k_edges, index = 1)
            loss_val += np.mean(np.abs(np.log(pk_[:,0]) - np.log(pk_ref[:,0])))
        print("Loss = ", loss_val, "params = ", params, flush = True)
        return loss_val
        
    if args.minimizer == "scipy":
            
        from scipy.optimize import minimize
        
        res = minimize(loss_fn, params, method='BFGS',
                options={'xatol': 1e-8, 'disp': True})
        
    elif args.minimizer == "nautilus":
        
        from nautilus import Prior, Sampler
        
        prior = Prior()
        prior.add_parameter('fac_1_exp', dist=(-2, 1))
        prior.add_parameter('rad_1', dist=(0, 20))
        prior.add_parameter('fac_2_exp', dist=(-2, 1))
        prior.add_parameter('rad_2', dist=(0, 20))
        prior.add_parameter('vel', dist=(0, 20))
        
        def likelihood(param_dict):
            params = np.array([10**param_dict['fac_1_exp'], param_dict['rad_1'],
                               10**param_dict['fac_2_exp'], param_dict['rad_2'],
                               param_dict['vel']])
            return -0.5 * loss_fn(params, fit_pk = True)
        
        
        sampler = Sampler(prior, likelihood, n_live=1000, filepath = f"data/fit_example_elg1.1.hdf5")
        sampler.run(verbose=True)
        
        
        
        
        
        
