import numpy as np
import jax 
import jax.numpy as jnp
import sys, os
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/cosmomia/")
from cosmomia import py_assign_particles_to_gals, subgrid_collapse
from mas import cic_mas_vec
from correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff
from displacements import enhance_short_range, interpolate_field
from astropy.table import Table, vstack
sys.path.append("/home/astro/dforero/codes//pyfcfc/")
from pyfcfc.boxes import py_compute_cf




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
    ax[1].semilogx(tpcf['s'], tpcf['multipoles'][0,0,:])
    
    
    cache_name = f"data/test_cache.npz"
    
    result = dict(np.load(cache_name))
    
    result['is_attractor'] = result['is_dm'].astype(bool) & (result['dweb'] < 3)
    
    print((result['is_attractor'].astype(bool)).sum() / result['is_attractor'].shape[0])
    
    
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
    ax[1].semilogx(tpcf['s'], tpcf['multipoles'][0,0,:])
    
    
    
    params = np.array([0.9, 1. * BIN_SIZE[0], 0.5, BIN_SIZE[0], 0.1], dtype = np.float32)
    result = subgrid_collapse(result, params, BOX_SIZE, result['is_attractor'].astype(bool), 99, debug = False)
    result['pos'] += BOX_SIZE
    result['pos'] %= BOX_SIZE
        
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
    ax[0].plot(k_, k_ * pk_, label = "CosmoMIA Coll")
    
    
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
    ax[1].semilogx(tpcf['s'], tpcf['multipoles'][0,0,:])
    
    
    ax[0].legend()
    ax[1].format(yscale = 'symlog')
    fig.savefig("plots/test_collapse.png", dpi=300)
    
    