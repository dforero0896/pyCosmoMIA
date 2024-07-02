import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import proplot as pplt
plt.style.use(['science', 'grid'])
pplt.rc.cycle = "bmh"
import jax
import jax.numpy as jnp
import jax_cosmo as jc
from jax_cosmo.power import linear_matter_power, nonlinear_matter_power
import sys, os, time
from cosmomia.displacements import aug_lpt, spherical_collapse, two_lpt, zeldovich, divergence_to_displacement, interpolate_field, two_lpt_, nbody, lpt_init, gen_ode_func, eul_aug_lpt, rank_order_fields, apply_transfer_func
from cosmomia.ics import linear_field_box_muller
from cosmomia.mas import cic_mas_vec
from cosmomia import cy_read_cic_vector, cy_cic_mas
from cosmomia.correlations import powspec_vec, naive_pk, naive_xpk, naive_rcoeff, compute_transfer_from_power, compute_transfer_from_cross_power
from pypowspec.pypowspec import compute_auto_box_mesh, compute_auto_box
from tqdm import tqdm
from mpi4py import MPI
comm = MPI.COMM_WORLD


interpolate_field_3d = jax.jit(jax.vmap(interpolate_field, in_axes = (0, None, None, None, None, None, None)))


box_size = [2000.] * 3
box_min = [0.] * 3
fig, ax = pplt.subplots([1])



def compute_alpt_reference(ic_field, chunked = False, window_size = None):
    N_big = ic_field.shape[0]
    
    delta_lin = ic_field
    mesh_shape = ic_field.shape
    bin_size = jnp.array(box_size) / jnp.array(mesh_shape)
    tic = time.time()
    psi = jnp.stack(aug_lpt(delta_lin, box_size[0], 0., 5, sc_alpha = 3./ 2), axis = 0)
    print(psi.shape)
    print(f"Ran ALPT sim in {time.time() - tic}s", flush=True)
    if not chunked:
        #pos_lagrangian = jnp.linspace(0, box_size[0], mesh_shape[0])
        pos_lagrangian = jnp.arange(0, box_size[0], bin_size[0])
        pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, mesh_shape[0] * mesh_shape[1] * mesh_shape[2]).T
        fname = f"data/multiscale/exact/complete/alpt_abacus_pos_N{N_big}.npy"
        new_pos = pos_lagrangian.copy()
        new_pos += interpolate_field_3d(psi, pos_lagrangian, 0,0,0, psi.shape[-1], box_size[0]).T
        
        np.save(fname, new_pos)
    else:
        ics_hr = ic_field
        assert not window_size is None
        bin_size = jnp.array(box_size) / jnp.array(mesh_shape)
        small_box_size = jnp.array(box_size) / (jnp.array(mesh_shape) // window_size)
        for ii, i in enumerate(range(0, ics_hr.shape[0], window_size)):
            for jj, j in enumerate(range(0, ics_hr.shape[1], window_size)):
                for kk, k in enumerate(range(0, ics_hr.shape[2], window_size)):
                    fname = f"data/multiscale/exact/chunked/alpt_abacus_pos_N{N_big}_{ii:04d}_{jj:04d}_{kk:04d}.npy"
                    ics = ics_hr[i:i+window_size, j:j+window_size, k:k+window_size]
                    
                    
                    xs = jnp.arange(0, small_box_size[0], bin_size[0]) + i * bin_size[0]
                    ys = jnp.arange(0, small_box_size[1], bin_size[1]) + j * bin_size[1]
                    zs = jnp.arange(0, small_box_size[2], bin_size[2]) + k * bin_size[2]
                    
                    
                    pos_lagrangian = jnp.array(jnp.meshgrid(xs, ys, zs)).reshape(3, ics.shape[0] * ics.shape[1] * ics.shape[2]).T
                    new_pos = pos_lagrangian.copy()
                    new_pos += interpolate_field_3d(psi, pos_lagrangian, 0,0,0, psi.shape[-1], box_size[0]).T
                    
                    
                    np.save(fname, new_pos)
        
    print(f"Simulation plus IO in {time.time() - tic}s", flush=True)
def apply_kernel(field, kernel):
    return jnp.fft.irfftn(jnp.fft.rfftn(field) * kernel, field.shape)

apply_kernel_3d = jax.jit(jax.vmap(apply_kernel, in_axes = (0, None)))


    

def alpt_subboxes(ics_lr, ics_hr, window_size, box_size, interp_smooth):
    import itertools
    mesh_shape = ics_hr.shape
    n_bins = ics_lr.shape[0]

    #pos_lagrangian = jnp.linspace(0, box_size[0], mesh_shape[0])
    #print(pos_lagrangian[:10])
    
    bin_size = jnp.array(box_size) / jnp.array(mesh_shape)
    small_box_size = jnp.array(box_size) / (jnp.array(mesh_shape) // n_bins)
    
    print(f"Computing long range contribution from LR ICs", flush = True)
    tic = time.time()
    psi_lr = jnp.stack(aug_lpt(ics_lr, box_size[0], 0., 5, sc_alpha = 3./ 2), axis = 0)
    #psi_lr = aug_lpt(ics_lr, box_size[0], 0., 5, sc_alpha = 3./ 2)
    
    print(f"Ran ALPT sim in {time.time() - tic}s", flush=True)
    print("Smoothing LR contribution", flush = True)
    k = jnp.fft.fftfreq(n_bins, d=box_size[0]/n_bins) * 2 * jnp.pi
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    ksq = ksq.at[0,0,0].set(0)
    kr = k * interp_smooth * 1.065
    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    #norm = 1 - jax.nn.sigmoid(50 *(jnp.sqrt(ksq) - 0.1))
    del k
    del ksq
    
    psi_lr = apply_kernel_3d(psi_lr, norm)
    #psi_lr = [apply_kernel(p, norm) for p in psi_lr]
     
    
    print(f"Computing short range contribution from HR ICs", flush=True)
        
    tic_ = time.time()
    #pos_lagrangian = jnp.array(jnp.meshgrid(pos_lagrangian, pos_lagrangian, pos_lagrangian)).reshape(3, jnp.prod(jnp.array(mesh_shape))).T
    #views = np.lib.stride_tricks.sliding_window_view(ics, (window_size,)*3, axis = (0,1,2))[::window_size,::window_size,::window_size]
    #for ii, i in enumerate(range(0, ics_hr.shape[0], window_size)):
    #    for jj, j in enumerate(range(0, ics_hr.shape[1], window_size)):
    #        for kk, k in enumerate(range(0, ics_hr.shape[2], window_size)):
    xs = jnp.arange(0, small_box_size[0], bin_size[0])
    ys = jnp.arange(0, small_box_size[1], bin_size[1])
    zs = jnp.arange(0, small_box_size[2], bin_size[2])
    
    
    k_ = jnp.fft.fftfreq(window_size, d=small_box_size[0]/window_size) * 2 * jnp.pi
    ksq = k_[:,None,None]**2 + k_[None,:,None]**2 + k_[None,None,:n_bins//2+1]**2
    ksq = ksq.at[0,0,0].set(0)
    #norm = jax.nn.sigmoid(50 * (jnp.sqrt(ksq) - 0.09))
    #norm = (1 + 1* jnp.sqrt(ksq)**2)
    kr = k_ * interp_smooth * 1.16 #This works with interp_smooth = 20.
    #kr = k_ * interp_smooth * 2  #This works with interp_smooth = 20.
    norm = 1 - jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :window_size//2+1] ** 2))
    
    
    
    my_chunks = np.array_split(list(itertools.product(range(0, ics_hr.shape[0], window_size), 
                                                                range(0, ics_hr.shape[1], window_size),
                                                                range(0, ics_hr.shape[2], window_size))),
                               comm.Get_size())[comm.Get_rank()]
    print(comm.Get_rank(), my_chunks.shape)
    
    
    
    for ii, (i, j, k) in enumerate(tqdm(my_chunks)):
        ics = ics_hr[i:i+window_size, j:j+window_size, k:k+window_size]
        
        fname = f"data/multiscale/approx/pos_approx_{i:04d}_{j:04d}_{k:04d}.npy"
        new_pos = simulate_subbox(ics, xs, ys, zs, (i,j,k), bin_size, psi_lr, small_box_size, norm)
        jnp.save(fname, new_pos)
    comm.Barrier()
    print(f"Ran all ALPT sim in {time.time() - tic_}s", flush=True)
@jax.jit
def simulate_subbox(ics, xs, ys, zs, indices, bin_size, psi_lr, small_box_size, norm):
    i, j, k = indices

    
    xs = xs.copy() + i * bin_size[0]
    ys = ys.copy() + j * bin_size[1]
    zs = zs.copy() + k * bin_size[2]
    
    
    pos_lagrangian = jnp.array(jnp.meshgrid(xs, ys, zs)).reshape(3, ics.shape[0] * ics.shape[1] * ics.shape[2]).T
    new_pos = pos_lagrangian.copy()
    
    new_pos += interpolate_field_3d(psi_lr, pos_lagrangian, 0,0,0, psi_lr.shape[-1], box_size[0]).T
    

    

    psi_hr = jnp.stack(aug_lpt(ics, small_box_size[0], 0., 5, sc_alpha = 3./ 2), axis = 0)
    
    

    psi_hr = apply_kernel_3d(psi_hr, norm)
    #new_pos += interpolate_field_3d(psi_hr, pos_lagrangian, xs[0], ys[0], zs[0], psi_hr.shape[-1], small_box_size[0]).T
    new_pos += interpolate_field_3d(psi_hr, pos_lagrangian, xs[0], ys[0], zs[0], psi_hr.shape[-1], small_box_size[0]).T
    
    return new_pos
    #print(f"{os.getpid()} for subbox({i},{j},{k}) done", flush = True)
                
def paint_to_mesh(particle_fns, mesh_shape, is_abacus = False):
    from tqdm import tqdm
    dtype = np.float32
    delta_ev = np.zeros(mesh_shape, dtype = dtype)
    
    _box_size = np.array(box_size, dtype = dtype)
    _box_min = np.array(box_min, dtype = dtype)
    
    for i, fn in enumerate(tqdm(sorted(particle_fns))):
        new_pos = np.load(fn).astype(dtype)
        if i == 0:
            w = np.ones(new_pos.shape[0], dtype = dtype)    
        if is_abacus:
            new_pos += 3 * box_size[0] / 2
            new_pos %= box_size[0]
        cy_cic_mas['float'](delta_ev, new_pos[:,0], new_pos[:,1], new_pos[:,2], w,
                        _box_min, _box_size, 1, 40)
        #delta_ev = cic_mas_vec(delta_ev,
        #                new_pos[:,0], new_pos[:,1], new_pos[:,2], jnp.broadcast_to(jnp.array([1.]), new_pos.shape[0]), 
        #                new_pos.shape[0], 
        #                0., 0., 0.,
        #                box_size[0],
        #                delta_ev.shape[0],
        #                True)
        #print(delta_ev[:10, 50, 50], flush = True)
        #print((delta_ev>=0).sum(), flush = True)
    #delta_ev /= delta_ev.mean()
    #delta_ev -= 1.
    
    return delta_ev

def read_particle_files(particle_fns, use_mp = True):
    from tqdm import tqdm
    if use_mp:
        pool = mp.Pool() 
        results = pool.map_async(np.load, particle_fns)
        pool.close()
        pool.join()
        results.wait()
        results = results.get()
    else:
        results = []
        for i, fn in enumerate(tqdm(sorted(particle_fns))):
            results.append(np.load(fn))
            
    return np.stack(results, axis = 0)
    
            
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", choices = ['reference' , 'split', 'clustering'], default = ['split', 'clustering'], nargs = "+")
    parser.add_argument("-rerun", action = "store_true")
    args = parser.parse_args()
    h = 0.6736
    cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.807952, n_s=0.9649,
                      Omega_k=0., w0=-1., wa=0.)


    D1 = jc.background.growth_factor(cosmo_jax, jnp.array([1./(1 + 1.1)]))
    print(D1, flush = True)
    if 'reference' in args.task:
        IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N1024_down.npy"
        ic_field = D1 * jnp.array(np.load(IC_FN))
        compute_alpt_reference(ic_field, chunked = True, window_size = 256)
        #compute_alpt_reference(ic_field, chunked = False, window_size = None)
        
        
        IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N256_down.npy"
        print(f"Loading LR ICs from {IC_FN}", flush = True)
        ics_lr = D1 * np.load(IC_FN)
        compute_alpt_reference(ics_lr, chunked = False, window_size = None)
    if 'split' in args.task :
        
        N = 256
        interp_smooth = 20.
        #interp_smooth = 0.1
        #IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N1024_down.npy"
        IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N1024_down.npy"
        print(f"Loading HR ICs from {IC_FN}", flush = True)
        #ics_hr = D1 * np.load(IC_FN)
        ics_hr = D1 * jnp.load(IC_FN, mmap_mode = 'r')
        #ics_hr = D1 * np.memmap(IC_FN, dtype = np.float32, shape = (1024, 1024, 1024), mode = 'r')
        IC_FN = f"/home/astro/dforero/cosmo3d/AbacusSummit/ics/ic_dens_N256_down.npy"
        print(f"Loading LR ICs from {IC_FN}", flush = True)
        ics_lr = D1 * np.load(IC_FN)
        alpt_subboxes(ics_lr, ics_hr, N, box_size, interp_smooth)
      
        
    if "clustering" in args.task and comm.Get_rank() == 0:
        import glob
        cmap = pplt.Colormap(('white', 'dark indigo'))
        VMIN = -0.1
        VMAX = 1.5
        fig, ax = pplt.subplots(nrows = 1, share = 0, refaspect = 2.6, figwidth = "12cm")
        pax = ax[:1].panel('bottom')
        
        
        
        
        mesh_shape = [1024] * 3
        k_in = 2 * jnp.pi / box_size[0]
        k_ny = jnp.mean(jnp.pi * jnp.array(mesh_shape) / jnp.array(box_size))
        k_edges = jnp.linspace(k_in, k_ny, 200)
        
        
        
        
        fname = f"data/multiscale/final/dmdens_abacus_alpt_exact_complete.npy"
        fname_pk = f"data/multiscale/final/pk_abacus_alpt_exact_complete.npy"
        if not os.path.isfile(fname) or not os.path.isfile(fname_pk):
            if not os.path.isfile(fname) or args.rerun:
                particle_fns = glob.glob("data/multiscale/exact/complete/alpt_abacus_pos_N1024.npy")
                #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
                #print(particles.shape)
                #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
                #k_, pk_ref = pk['k'], pk['multipoles'][:,0]
                delta_ev = paint_to_mesh(particle_fns, mesh_shape)
                np.save(fname, delta_ev)
            else:
                delta_ev = np.load(fname)
            if not os.path.isfile(fname_pk)  or args.rerun:
                #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
                pk = compute_auto_box_mesh(delta_ev.astype(np.double), "tests/powspec.conf")
                np.save(fname_pk, pk)
            else:
                pk = np.load(fname_pk, allow_pickle=True).item()
        else:
            delta_ev = np.load(fname)
            pk = np.load(fname_pk, allow_pickle=True).item()
        #k_, pk_ref = naive_pk(delta_ev, box_size[0], k_edges)  
        k_, pk_ref = pk['k'], pk['multipoles'][:,0]
        ax[0].loglog(k_, pk_ref, label = "ALPT $N_{\mathrm{big}}^3$")
        pax[0].plot(k_, pk_ref/pk_ref - 1)
        #ax[1].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        fig.savefig("plots/example_multiscale.png", dpi=300)
        
        
        
        fname = f"data/multiscale/final/dmdens_abacus_alpt_exact_complete_small.npy"
        fname_pk = f"data/multiscale/final/pk_abacus_alpt_exact_complete_small.npy"
        if not os.path.isfile(fname) or not os.path.isfile(fname_pk):
            if not os.path.isfile(fname)  or args.rerun:
                particle_fns = glob.glob("data/multiscale/exact/complete/alpt_abacus_pos_N256.npy")
                #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
                #print(particles.shape)
                #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
                #k_, pk_ = pk['k'], pk['multipoles'][:,0]
                delta_ev = paint_to_mesh(particle_fns, mesh_shape)
                np.save(fname, delta_ev)
            else:
                delta_ev = np.load(fname)
            if not os.path.isfile(fname_pk)  or args.rerun:
                #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
                pk = compute_auto_box_mesh(delta_ev.astype(np.double), "tests/powspec.conf")
                np.save(fname_pk, pk)
            else:
                pk = np.load(fname_pk, allow_pickle=True).item()
        else:
            delta_ev = np.load(fname)
            pk = np.load(fname_pk, allow_pickle=True).item()
        k_, pk_ = pk['k'], pk['multipoles'][:,0]
        ax[0].loglog(k_, pk_, label = "ALPT $N_{\mathrm{small}}^3$")
        pax[0].plot(k_, pk_ / pk_ref - 1,)
        #ax[2].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        fig.savefig("plots/example_multiscale.png", dpi=300)
        
        
        
        
        
        #fname = f"data/multiscale/final/dmdens_abacus_alpt_exact_chunked.npy"
        #if not os.path.isfile(fname)  or args.rerun:# :
        #    particle_fns = glob.glob("data/multiscale/exact/chunked/alpt*npy")
        #    #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
        #    #print(particles.shape)
        #    #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
        #    #k_, pk_ = pk['k'], pk['multipoles'][:,0]
        #    delta_ev = paint_to_mesh(particle_fns, mesh_shape)
        #    np.save(fname, delta_ev)
        #else:
        #    delta_ev = np.load(fname)
        ##k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
        #pk = compute_auto_box_mesh(delta_ev.astype(np.double), "tests/powspec.conf")
        #k_, pk_ = pk['k'], pk['multipoles'][:,0]
        #ax[0].loglog(k_, pk_, label = "Exact Chunk.")
        #pax[0].plot(k_, pk_ / pk_ref - 1,)
        #fig.savefig("plots/example_multiscale.png", dpi=300)
        
        
        
        fname = f"data/reference/dmdens_abacus.npy"
        fname_pk = f"data/reference/pk_abacus.npy"
        if not os.path.isfile(fname) or not os.path.isfile(fname_pk) or args.rerun:
            if not os.path.isfile(fname) or args.rerun:
                particle_fns = glob.glob("/srv/astro/projects/cosmo3d/AbacusSummit/ics/dm_part_N512_z1.100.npy")
                #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
                #print(particles.shape)
                #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
                #k_, pk_ = pk['k'], pk['multipoles'][:,0]
                delta_ev_abacus = paint_to_mesh(particle_fns, mesh_shape, is_abacus = True)
                np.save(fname, delta_ev_abacus)
            else:
                delta_ev_abacus = np.load(fname)
            if not os.path.isfile(fname_pk)  or args.rerun:
                #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
                pk = compute_auto_box_mesh(delta_ev_abacus.astype(np.double), "tests/powspec.conf")
                np.save(fname_pk, pk)
            else:
                pk = np.load(fname_pk, allow_pickle=True).item()
        else:
            delta_ev_abacus = np.load(fname)
            pk = np.load(fname_pk, allow_pickle=True).item()
        #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
        k_, pk_ = pk['k'], pk['multipoles'][:,0]
        ax[0].loglog(k_, pk_, label = "Abacus")
        pax[0].loglog(k_, pk_ / pk_ref - 1,)
        
        
        fname = f"data/multiscale/final/dmdens_abacus_alpt_approx.npy"
        fname_pk = f"data/multiscale/final/pk_abacus_alpt_approx.npy"
        if not os.path.isfile(fname) or not os.path.isfile(fname_pk)  or 'split' in args.task  or args.rerun :
            if not os.path.isfile(fname) or 'split' in args.task  or args.rerun :
                particle_fns = glob.glob("data/multiscale/approx/pos*npy")
                #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
                #print(particles.shape)
                #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
                #k_, pk_ = pk['k'], pk['multipoles'][:,0]
                delta_ev = paint_to_mesh(particle_fns, mesh_shape)
                np.save(fname, delta_ev)
            else:
                delta_ev = np.load(fname)
            if not os.path.isfile(fname_pk) or 'split' in args.task or args.rerun :
                #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
                pk = compute_auto_box_mesh(delta_ev.astype(np.double), "tests/powspec.conf")
                np.save(fname_pk, pk)
            else:
                pk = np.load(fname_pk, allow_pickle=True).item()
        else:
            delta_ev = np.load(fname)
            pk = np.load(fname_pk, allow_pickle=True).item()
        #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
        k_, pk_ = pk['k'], pk['multipoles'][:,0]
        ax[0].loglog(k_, pk_, label = "LANA $N_{\mathrm{big}}^3*N_{\mathrm{small}}^3$")
        pax[0].plot(k_, pk_ / pk_ref - 1,)
        #ax[3].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        
        
        fig.savefig("plots/example_multiscale.png", dpi=300)
        
        
        
        
        k_edges_interp = jnp.linspace(k_in, jnp.sqrt(3) * k_ny, 200)
        
        
        fname = f"data/multiscale/final/dmdens_abacus_alpt_remap.npy"
        fname_pk = f"data/multiscale/final/pk_abacus_alpt_remap.npy"
        if not os.path.isfile(fname) or not os.path.isfile(fname_pk)  or 'split' in args.task  or args.rerun or 1:
            if not os.path.isfile(fname) or 'split' in args.task  or args.rerun :
                
                #particles = (read_particle_files(particle_fns, use_mp = False) + box_size) % box_size
                #print(particles.shape)
                #pk = compute_auto_box(particles[0,:,0],particles[0,:,1], particles[0,:,2], np.ones_like(particles[0,:,2]), "tests/powspec.conf")
                #k_, pk_ = pk['k'], pk['multipoles'][:,0]
                k_, transfer = compute_transfer_from_power(delta_ev, delta_ev_abacus, box_size[0], k_edges_interp)
                assert (~np.isnan(transfer)).all()
                delta_ev = apply_transfer_func(rank_order_fields(delta_ev, apply_transfer_func(delta_ev_abacus, box_size[0], transfer, k_), box_size[0],1.), box_size[0], 1 / transfer, k_)
                np.save(fname, delta_ev)
            else:
                delta_ev = np.load(fname)
            if not os.path.isfile(fname_pk) or 'split' in args.task or args.rerun or 1:
                #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
                pk = compute_auto_box_mesh(np.array(delta_ev).astype(np.double), "tests/powspec.conf")
                np.save(fname_pk, pk)
            else:
                pk = np.load(fname_pk, allow_pickle=True).item()
        else:
            delta_ev = np.load(fname)
            pk = np.load(fname_pk, allow_pickle=True).item()
        #k_, pk_ = naive_pk(delta_ev, box_size[0], k_edges)  
        k_, pk_ = pk['k'], pk['multipoles'][:,0]
        ax[0].loglog(k_, pk_, label = "LANARM $N_{\mathrm{big}}^3*N_{\mathrm{small}}^3$")
        pax[0].plot(k_, pk_ / pk_ref - 1,)
        #ax[3].imshow(jnp.log10(2 + delta_ev[:20, :, :].mean(axis=0)), colorbar = None, vmin = VMIN, cmap = cmap, vmax = VMAX)
        
        
        fig.savefig("plots/example_multiscale.png", dpi=300)
        
        
        ax[0].legend(loc = 'top')
        pax[0].set_yscale('symlog', linthresh = 0.1)
        pax[0].format(ylim = (-0.1, 0.1), ylocator = (-0.05, 0, 0.05))
        pax[0].fill_between(k_, -0.05, 0.05, c = 'gray5', zorder = 0)
        ax[0].format(yformatter = 'log', ylabel = "$P(k)$", xlabel = "$k~[h/\mathrm{Mpc}]$", ylim = (1e1, 1e4))
        pax[0].format(ylabel = "$\Delta P(k) / P_{N_\mathrm{big}}$",)
        fig.savefig("plots/example_multiscale.png", dpi=300)
        