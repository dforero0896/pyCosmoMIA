import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator
from .mas import cic_mas_vec
import jax_cosmo as jc





def get_k(n_bins, box_size):
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    return k

@jax.jit
def vel_kernel(field, gain, box_size):
    
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    k = jnp.sqrt(k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2)
    field_k = jnp.fft.rfftn(field)
    field_k *= 1 + jax.nn.sigmoid(0.6) * gain * k**2
    field = jnp.fft.irfftn(field_k, field.shape)
    return field


@jax.jit
def divergence_to_displacement(divergence, box_size, smooth):

    n_bins = divergence.shape[0]
    phi_field = jnp.fft.rfftn(divergence)
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)

    psi_x = jnp.fft.irfftn(-1j * k[:,None, None] * phi_field, divergence.shape)
    psi_y = jnp.fft.irfftn(-1j * k[None,:,None] * phi_field, divergence.shape)
    psi_z = jnp.fft.irfftn(-1j * k[None,None,:n_bins//2 + 1] * phi_field, divergence.shape)

    return psi_x, psi_y, psi_z


@jax.jit
def divergence_to_displacement_k(phi_field, box_size, smooth):

    n_bins = phi_field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)

    psi_x = jnp.fft.irfftn(-1j* k[:,None, None] * phi_field, (n_bins, n_bins, n_bins))
    psi_y = jnp.fft.irfftn(-1j* k[None,:,None] * phi_field, (n_bins, n_bins, n_bins))
    psi_z = jnp.fft.irfftn(-1j* k[None,None,:n_bins//2 + 1] * phi_field, (n_bins, n_bins, n_bins))

    return psi_x, psi_y, psi_z

@jax.jit
def zeldovich(delta, box_size, smooth):

    divergence = - delta

    psi_x, psi_y, psi_z = divergence_to_displacement(divergence, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def two_lpt(delta, box_size, smooth):

    """ See https://github.com/DifferentiableUniverseInitiative/flowpm/blob/8f9415be8866f507fe40259c4dd3fc981583ee41/flowpm/tfpm.py#L98"""
    
    n_bins = delta.shape[0]
    phi_field = jnp.fft.rfftn(delta)
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)
    
    phi_field_2 = jnp.zeros_like(delta) #Stores laplacian of Phi_2 first
    
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]
    
    
    phi_ii = []
    phi_ii.append(jnp.fft.irfftn((-k[:,None,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,:,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]**2) * phi_field, delta.shape))
    
    phi_field_2 += phi_ii[1] * phi_ii[2]
    phi_field_2 += phi_ii[2] * phi_ii[0]
    phi_field_2 += phi_ii[0] * phi_ii[1]
    
    del phi_ii
    
       
    phi_field_2 -= (jnp.fft.irfftn((-k[None,:,None]*k[None,None,:n_bins // 2 + 1]) * phi_field, delta.shape))**2
    phi_field_2 -= (jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]*k[:,None,None]) * phi_field, delta.shape))**2
    phi_field_2 -= (jnp.fft.irfftn((-k[:,None,None]*k[None,:,None]) * phi_field, delta.shape))**2
    
    phi_field_2 *= 3. / 7
    # Solve Poisson eq for Phi_2 in Fourier space
    phi_field_2 = jnp.fft.rfftn(phi_field_2)
    phi_field_2 = jnp.where(ksq > 0., phi_field_2 * norm / (ksq), 0.)
    phi_field_2 = phi_field_2.at[0,0,0].set(0.)
    
    # Compute first order displacements
    psi_x = jnp.fft.irfftn(-1j* k[:,None, None] * -phi_field, (n_bins, n_bins, n_bins))
    psi_y = jnp.fft.irfftn(-1j* k[None,:,None] * -phi_field, (n_bins, n_bins, n_bins))
    psi_z = jnp.fft.irfftn(-1j* k[None,None,:n_bins//2 + 1] * -phi_field, (n_bins, n_bins, n_bins))
    
    # Compute second order displacements
    psi_x -= jnp.fft.irfftn(-1j* k[:,None, None] * phi_field_2, (n_bins, n_bins, n_bins))
    psi_y -= jnp.fft.irfftn(-1j* k[None,:,None] * phi_field_2, (n_bins, n_bins, n_bins))
    psi_z -= jnp.fft.irfftn(-1j* k[None,None,:n_bins//2 + 1] * phi_field_2, (n_bins, n_bins, n_bins))

    return psi_x, psi_y, psi_z

@jax.jit
def two_lpt_(delta, box_size, smooth):

    """ See https://github.com/DifferentiableUniverseInitiative/flowpm/blob/8f9415be8866f507fe40259c4dd3fc981583ee41/flowpm/tfpm.py#L98"""
    
    n_bins = delta.shape[0]
    phi_field = jnp.fft.rfftn(delta)
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2

    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)
    
    phi_field_2 = jnp.zeros_like(delta) #Stores laplacian of Phi_2 first
    
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]
    
    
    phi_ii = []
    phi_ii.append(jnp.fft.irfftn((-k[:,None,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,:,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]**2) * phi_field, delta.shape))
    
    phi_field_2 += phi_ii[1] * phi_ii[2]
    phi_field_2 += phi_ii[2] * phi_ii[0]
    phi_field_2 += phi_ii[0] * phi_ii[1]
    
    del phi_ii
    
       
    phi_field_2 -= (jnp.fft.irfftn((-k[None,:,None]*k[None,None,:n_bins // 2 + 1]) * phi_field, delta.shape))**2
    phi_field_2 -= (jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]*k[:,None,None]) * phi_field, delta.shape))**2
    phi_field_2 -= (jnp.fft.irfftn((-k[:,None,None]*k[None,:,None]) * phi_field, delta.shape))**2
    
    divergence_k = -ksq * phi_field + (-3. / 7) * jnp.fft.rfftn(phi_field_2)
    psi_x, psi_y, psi_z = divergence_to_displacement_k(divergence_k, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def spherical_collapse(delta, box_size, smooth, alpha = 3. / 2):

    """ See eq 2.13 https://florent-leclercq.eu/documents/thesis/Chapter2.pdf"""

    divergence = jnp.where(delta < alpha, 3 * ((1. - delta / alpha)**(alpha / 3) - 1.), -3.)
    
    psi_x, psi_y, psi_z = divergence_to_displacement(divergence, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def aug_lpt(delta, box_size, smooth, interp_smooth, sc_alpha = 3./2):

    """ See eq 2.17 https://florent-leclercq.eu/documents/thesis/Chapter2.pdf"""

    n_bins = delta.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * smooth
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    #norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    norm = 1
    kr = k * interp_smooth
    interp_kernel = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))


    divergence_sc_k = jnp.fft.rfftn(jnp.where(delta < sc_alpha, 3 * ((1. - delta / sc_alpha)**(sc_alpha / 3) - 1.), -3.))
    #divergence_2lpt_k = jnp.fft.rfftn(-delta + (3. / 7) * delta**2)
    
    # Solve Poisson for Phi_1
    phi_field = jnp.fft.rfftn(delta)
    phi_field = jnp.where(ksq > 0., phi_field * norm / (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)
    lap_phi_field_2 = jnp.zeros_like(delta) #Stores laplacian of Phi_2 first
    
    
    D1 = [1, 2, 0]
    D2 = [2, 0, 1]
    
    
    phi_ii = []
    phi_ii.append(jnp.fft.irfftn((-k[:,None,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,:,None]**2) * phi_field, delta.shape))
    phi_ii.append(jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]**2) * phi_field, delta.shape))
    
    lap_phi_field_2 += phi_ii[1] * phi_ii[2]
    lap_phi_field_2 += phi_ii[2] * phi_ii[0]
    lap_phi_field_2 += phi_ii[0] * phi_ii[1]
    
    del phi_ii
    
       
    lap_phi_field_2 -= (jnp.fft.irfftn((-k[None,:,None]*k[None,None,:n_bins // 2 + 1]) * phi_field, delta.shape))**2
    lap_phi_field_2 -= (jnp.fft.irfftn((-k[None,None,:n_bins // 2 + 1]*k[:,None,None]) * phi_field, delta.shape))**2
    lap_phi_field_2 -= (jnp.fft.irfftn((-k[:,None,None]*k[None,:,None]) * phi_field, delta.shape))**2
    
    
    
    divergence_2lpt_k = -ksq * phi_field + (-3. / 7) * jnp.fft.rfftn(lap_phi_field_2)
    
    divergence_k = interp_kernel * divergence_2lpt_k + (1. - interp_kernel) * divergence_sc_k
    #divergence_k = -(ksq) * phi_field
    #divergence_k = divergence_2lpt_k
    
    psi_x, psi_y, psi_z = divergence_to_displacement_k(divergence_k, box_size, smooth)

    return psi_x, psi_y, psi_z

@jax.jit
def rank_order_fields(field, ref_field, box_size, smooth):
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    ksq = ksq.at[0,0,0].set(0)
    kr = k * smooth
    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    
    field_sorter = jnp.argsort(jnp.fft.irfftn(norm * jnp.fft.rfftn(field), field.shape).ravel())[::-1]
    ref_field_sorter = jnp.argsort(jnp.fft.irfftn(norm * jnp.fft.rfftn(ref_field), ref_field.shape).ravel())[::-1]
    field = field.ravel().at[field_sorter].set(ref_field.ravel()[ref_field_sorter])
    
    return field.reshape(ref_field.shape)

@jax.jit
def correct_large_scale_bias(field, ref_field, box_size, smooth):
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    ksq = ksq.at[0,0,0].set(0)
    kr = k * smooth
    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    factor = jnp.mean(jnp.fft.irfftn(norm * jnp.fft.rfftn(ref_field), ref_field.shape) / jnp.fft.irfftn(norm * jnp.fft.rfftn(field), field.shape))
    return factor * field


def apply_transfer_func(field, box_size, transfer, k_transfer):
    print(transfer.shape, k_transfer.shape)
    transfer_func = lambda __k: RegularGridInterpolator((jnp.log(k_transfer),), transfer, method='linear', fill_value = 1., bounds_error = False)(jnp.log(__k))
    
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    k = jnp.sqrt(k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2)
    field_k = jnp.fft.rfftn(field)
    return jnp.fft.irfftn(field_k.at[:].set((transfer_func(k.ravel()).reshape(field_k.shape) * field_k)), field.shape)
    


@jax.jit
def add_interpolated_disp(pos, psi, box_min, box_size, fac = 1.):
    xmin, ymin, zmin = box_min
    for i in range(3):
        pos = pos.at[:,i].add(fac * interpolate_field(psi[i], pos, xmin, ymin, zmin, psi[i].shape[0], box_size[i]))
    return pos

def eul_aug_lpt(delta, box_size, smooth, interp_smooth, steps, pos, box_min, cosmo):
    xmin, ymin, zmin = box_min
    pos0 = pos.copy()
    D1_steps = jc.background.growth_factor(cosmo, steps)
    print(D1_steps)
    delta_D = D1_steps[0]
    delta_a = jnp.interp(delta_D, D1_steps, steps)
    D1 = jc.background.growth_factor(cosmo, jnp.atleast_1d(delta_a))
    pos = add_interpolated_disp(pos0, aug_lpt(delta_D * delta, box_size[0], smooth, interp_smooth), box_min, box_size)
    
    
    for i in range(1, len(D1_steps)):
        
        delta = delta.at[:].set(0.)
        delta = cic_mas_vec(delta,
                        pos[:,0], pos[:,1], pos[:,2], jnp.broadcast_to(1., pos.shape[0]), 
                        pos.shape[0], 
                        xmin, ymin, zmin,
                        box_size[0],
                        delta.shape[0],
                        True)
        
        delta /= delta.mean()
        delta -= 1.
        delta_D = (D1_steps[i])
        delta_a = jnp.interp(delta_D, D1_steps, steps)
        D1 = jc.background.growth_factor(cosmo, jnp.atleast_1d(delta_a))
        pos = add_interpolated_disp(pos0, aug_lpt(delta / D1_steps[i-1], box_size[0], smooth, interp_smooth), box_min, box_size, fac = delta_D)

    return pos

#@jax.jit
def enhance_short_range(disp, box_size, interp_smooth):
    
    n_bins = disp.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * interp_smooth
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    #interp_kernel = 1 - jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    interp_kernel = jax.scipy.special.erf((jnp.sqrt(kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1]**2)))
    assert jnp.all(~jnp.isnan(interp_kernel))
    return jnp.fft.irfftn((interp_kernel) * jnp.fft.rfftn(disp), disp.shape)
    #return jnp.fft.irfftn((jax.nn.sigmoid(100 * (jnp.sqrt(ksq) - interp_smooth))) * jnp.fft.rfftn(disp), disp.shape)


@jax.jit
def inverse_laplace_kernel(disp, box_size, interp_smooth):
    n_bins = disp.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins//2+1]**2
    ksq = ksq.at[0,0,0].set(0)
    kr = k * interp_smooth
    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    #norm = 1
    #return jnp.fft.irfftn(jnp.where(ksq > 0, 1 / ksq, 0), disp.shape)
    phi_field = jnp.where(ksq > 0., jnp.fft.rfftn(disp) * norm * (ksq), 0.)
    phi_field = phi_field.at[0,0,0].set(0.)
    return jnp.fft.irfftn(phi_field, disp.shape)
@jax.jit
def interpolate_field(disp, positions, xmin, ymin, zmin, n_bins, box_size):

    bin_size = box_size / n_bins
    xpos = (positions[:,0] - xmin) / bin_size
    ypos = (positions[:,1] - ymin) / bin_size
    zpos = (positions[:,2] - zmin) / bin_size

    i = jnp.int32(xpos)
    j = jnp.int32(ypos)
    k = jnp.int32(zpos)

    ddx = xpos - i
    ddy = ypos - j
    ddz = zpos - k

    def weights(ddx, ddy, ddz, ii, jj, kk):
        return (((1 - ddx) + ii * (-1 + 2 * ddx)) * 
                ((1 - ddy) + jj * (-1 + 2 * ddy)) *
                ((1 - ddz) + kk * (-1 + 2 * ddz)))

    shifts = jnp.zeros((positions.shape[0]))

    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                shifts = shifts.at[:].add(weights(ddx, ddy, ddz, ii, jj, kk) * disp[(i + ii) % n_bins, (j + jj) % n_bins, (k + kk) % n_bins])            

    
    return shifts

interpolate_field_3d = jax.jit(jax.vmap(interpolate_field, in_axes = (0, None, None, None, None, None, None)))

def weights(ddx, ddy, ddz, ii, jj, kk):
    return (((1 - ddx) + ii * (-1 + 2 * ddx)) * 
            ((1 - ddy) + jj * (-1 + 2 * ddy)) *
            ((1 - ddz) + kk * (-1 + 2 * ddz)))
    
def interpolate_field_single(disp, x, y, z, xmin, ymin, zmin, n_bins, box_size):

    bin_size = box_size / n_bins
    xpos = (x - xmin) / bin_size
    ypos = (y - ymin) / bin_size
    zpos = (z - zmin) / bin_size

    i = jnp.int32(xpos)
    j = jnp.int32(ypos)
    k = jnp.int32(zpos)

    ddx = xpos - i
    ddy = ypos - j
    ddz = zpos - k

    shifts = 0

    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                shifts += weights(ddx, ddy, ddz, ii, jj, kk) * disp[(i + ii) % n_bins, (j + jj) % n_bins, (k + kk) % n_bins]
                

    
    return shifts



def gaussian_field(grid, kf, Pkf, Rayleigh_sampling, 
                      key, box_size):
    
    """ Taken from https://github.com/bsciolla/gaussian-random-fields"""
    
    phase_prefac = 2.0*jnp.pi
    k_prefac     = 2.0*jnp.pi/box_size
    inv_max_rand = 1.0
    zero = 0. + 1j * 0.
    dims = grid
    k_bins, middle = kf.shape[0], grid//2

    # define the density field in Fourier space
    delta_k = jnp.zeros((grid, grid, grid), dtype=jnp.complex64)
    
    key_real, key_imag = jax.random.split(key)
    
    
    dims_range = jnp.arange(dims)
    
    ki = jnp.where(dims_range > middle, dims_range - dims, dims_range)
    
    
    kx = jnp.broadcast_to(ki[:, None, None], (dims, dims, dims))
    ky = jnp.broadcast_to(ki[None, :, None], (dims, dims, dims))
    kz = jnp.broadcast_to(ki[None, None, :], (dims, dims, dims))
    
    
    kmod = jnp.sqrt(kx**2 + ky**2 + kz**2) * k_prefac
    
    
    Pk = jnp.interp(kmod.flatten(), kf, Pkf).reshape(delta_k.shape)#
    
    Pk *= (grid**2/box_size)**3 
    
    amplitude = jnp.sqrt(Pk)   
    amplitude = jax.ops.index_update(amplitude, jax.ops.index[0,0,0], 0.)
    real_part = jax.random.normal(key_real, shape = (grid, grid, grid))
    imag_part = jax.random.normal(key_imag, shape = (grid, grid, grid))
    phase = jnp.arctan(imag_part / real_part)
    noise = real_part + 1j * imag_part
    #noise = (real_part**2 + imag_part**2)**0.5*jnp.exp(1j * phase)
    
    delta = jnp.fft.ifftn(noise * amplitude).real
    
    
    return delta


def get_positions(key, number_particles, bin_size):
    key, subkey = jax.random.split(key)
    Rs = 2 * jax.random.uniform(subkey, shape = (number_particles,3)) - 1
    Rs = jnp.sign(Rs) * (1 - jnp.sqrt(jnp.abs(Rs)))
    
    return Rs * bin_size

def populate_field(rho, n_bins, box_size, density, key):
    bin_size = box_size / n_bins
    cell_volume = bin_size**3
    mean_obj_per_cell = cell_volume * density
    
    rho *= mean_obj_per_cell / rho.mean()
    nonzero = (rho != 0).sum()
    sorted_rho = jnp.argsort(rho.ravel())[::-1]#[:nonzero+1]
    
    flat_rho = rho.flatten()[sorted_rho]
    #grid_centers = jnp.arange(0, box_size, box_size / n_bins) + 0.5 * box_size / n_bins
    #grid_centers = jnp.array(jnp.meshgrid(grid_centers, grid_centers, grid_centers)).reshape(3, n_bins**3).T
    grid_centers = jnp.array(jnp.unravel_index(sorted_rho, (n_bins, n_bins, n_bins))).T.astype(jnp.float32) * bin_size + 0.5 * bin_size
    
    
    key, subkey = jax.random.split(key)
    number_objects = jax.random.poisson(subkey, flat_rho, shape=flat_rho.shape)
    print(number_objects.sum())
    
    displacements = get_positions(key, number_objects.sum(), bin_size)

    coords = jnp.repeat(grid_centers, number_objects, axis=0) + displacements
    return (coords + box_size) % box_size


def field_smooth(field, scale, box_size):
    n_bins = field.shape[0]
    k = jnp.fft.fftfreq(n_bins, d=box_size/n_bins) * 2 * jnp.pi
    kr = k * scale

    norm = jnp.exp(-0.5 * (kr[:, None, None] ** 2 + kr[None, :, None] ** 2 + kr[None, None, :n_bins//2+1] ** 2))
    
    return jnp.fft.irfftn(norm * jnp.fft.rfftn(field), field.shape)







# JAX PM #buggy, adapted from flowpm
def inv_laplacian_k(k):
    n_bins = k.shape[0]
    ksq = k[:,None,None]**2 + k[None,:,None]**2 + k[None,None,:n_bins // 2 + 1]**2
    return jnp.where(ksq == 0., 1., 1. / ksq)

def gradient_ki(ki, order):
    
    def order_0_fun(ki):
        return 1j * ki
    def order_not_0_fun(ki):
        a = 1 / 6.0 * (8 * jnp.sin(ki) - jnp.sin(2 * ki))
        return 1j * a
    
    return jax.lax.cond(order == 0, order_0_fun, order_not_0_fun, ki)

def long_range_k(ksq, r_split):
    
    return jax.lax.cond(r_split==0, lambda ksq, r: jnp.broadcast_to(1., ksq.shape), lambda ksq, r: jnp.exp(- ksq * r**2), ksq, r_split)

@jax.jit
def cic_interpolate(field, particles, box_size, n_bins):
    
    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1], [1., 1, 0],
                      [1., 0, 1], [0., 1, 1], [1., 1, 1]]]).squeeze()
    floor = jnp.floor(particles) * n_bins // box_size
    part = particles * n_bins / box_size
    neighbors = floor[:,None,:] + connection[None,...]
    kernel = 1. - jnp.abs(part[:,None,:] - neighbors)    
    kernel = kernel[...,0] * kernel[...,1] * kernel[...,2]
    neighbors = neighbors.astype(jnp.int32)
    neighbors %= n_bins
    field_eval = field[neighbors[...,0].reshape(-1), neighbors[...,1].reshape(-1), neighbors[...,2].reshape(-1)].reshape(-1, 8) * kernel
    return field_eval.sum(axis=-1)

@jax.jit
def apply_longrange(positions,
                    delta_k,
                    box_size,
                    split,
                    factor,
                    ki):
    
    n_bins = delta_k.shape[0]
    ksq = ki[:,None,None]**2 + ki[None,:,None]**2 + ki[None,None,:n_bins//2+1]**2
    inv_laplace = inv_laplacian_k(ki)
    smoothing = long_range_k(ksq, split)
    pot_k = delta_k * inv_laplace * smoothing
    forces = jnp.empty_like(positions)
    
    forces = forces.at[:,0].set(cic_interpolate(jnp.fft.irfftn(pot_k * gradient_ki(ki[:,None,None], 0), (n_bins, n_bins, n_bins)), positions, box_size, n_bins))
    forces = forces.at[:,1].set(cic_interpolate(jnp.fft.irfftn(pot_k * gradient_ki(ki[None,:,None], 0), (n_bins, n_bins, n_bins)), positions, box_size, n_bins))
    forces = forces.at[:,2].set(cic_interpolate(jnp.fft.irfftn(pot_k * gradient_ki(ki[None,None,:n_bins//2 + 1], 0), (n_bins, n_bins, n_bins)), positions, box_size, n_bins))
    
    return forces * factor
    
    
@jax.jit    
def force(cosmo, particles, rho, box_size):
    
    n_bins = rho.shape[0]
    rho.at[...].set(0.)
    w0 = jnp.broadcast_to(1., particles.shape[0])
    rho = cic_mas_vec(rho,
                    particles[:,0], particles[:,1], particles[:,2], w0, 
                    particles.shape[0], 
                    0., 0., 0.,
                    box_size,
                    n_bins,
                    True)
    rho /= rho.mean()
    ki = get_k(n_bins, box_size)
    delta_k = jnp.fft.rfftn(rho)
    fac = 1.5 * cosmo.Omega_m
    forces = apply_longrange(particles, delta_k, box_size, 0., fac, ki)
        
    return forces
    
@jax.jit
def kick(cosmo, forces, ai, ac, af, **kwargs):
    Ei = jnp.sqrt(jc.background.Esqr(cosmo, jnp.atleast_1d(ai)))
    Ef = jnp.sqrt(jc.background.Esqr(cosmo, jnp.atleast_1d(af)))
    Ec = jnp.sqrt(jc.background.Esqr(cosmo, jnp.atleast_1d(ac)))
    D1fi = jc.background.growth_rate(cosmo, jnp.atleast_1d(ai)) / ai
    D1ff = jc.background.growth_rate(cosmo, jnp.atleast_1d(af)) / af
    D1fc = jc.background.growth_rate(cosmo, jnp.atleast_1d(ac)) / ac
    
    D1c = jc.background.growth_factor(cosmo, jnp.atleast_1d(ac))
    
    
    Omega_m_ac = jc.background.Omega_m_a(cosmo, jnp.atleast_1d(ac))
    Omega_de_ac = jc.background.Omega_de_a(cosmo, jnp.atleast_1d(ac))
    
    F1pc = 1.5 * Omega_m_ac * D1c / ac**2 - (D1fc / ac) * (
      Omega_de_ac - 0.5 * Omega_m_ac + 2) / jc.background.growth_factor(cosmo, jnp.atleast_1d(1.))[0]
    
    Gfi = D1fi * ai**3 * Ei
    Gff = D1ff * af**3 * Ef
    
    fdec = jc.background.f_de(cosmo, jnp.atleast_1d(ac))
    epsilon = 1e-5
    dfdec = (3 * cosmo.wa * (jnp.log(ac - epsilon) - (ac - 1) / (ac - epsilon)) /
          jnp.power(jnp.log(ac - epsilon), 2))
    
    dEac = 0.5 * (-3 * cosmo.Omega_m / ac**4 -
                2 * cosmo.Omega_k / ac**3 + dfdec *
                cosmo.Omega_de * jnp.power(ac, fdec)) / Ec
    
    gfc = (F1pc * ac**3 * Ec + D1fc * ac**3 * dEac +
          3 * ac**2 * Ec * D1fc)
    fac = 1 / (ac**2 * Ec) * (Gff - Gfi) / gfc
    
    return fac * forces

@jax.jit
def drift(cosmo, momenta, ai, ac, af):
  
    
    Ec = jnp.sqrt(jc.background.Esqr(cosmo, jnp.atleast_1d(ac)))
    D1fc = jc.background.growth_rate(cosmo, jnp.atleast_1d(ac)) / ac
    
    D1i = jc.background.growth_factor(cosmo, jnp.atleast_1d(ai))
    D1f = jc.background.growth_factor(cosmo, jnp.atleast_1d(af))
    fac = 1. / (ac**3 * Ec) * (D1f - D1i) / D1fc
    
    
    return fac * momenta


    
@jax.jit
def lpt_init(cosmo, linear, a, pos_lagrangian, box_size, n_bins):
    
    state = {}
    
        
    D1 = jc.background.growth_factor(cosmo, jnp.atleast_1d(a))[0]
    f1 = jc.background.growth_rate(cosmo, jnp.atleast_1d(a))[0]
    E = jc.background.Esqr(cosmo, jnp.atleast_1d(a))[0]**0.5
    Omega_m_a = jc.background.Omega_m_a(cosmo, jnp.atleast_1d(a))
    Omega_de_a = jc.background.Omega_de_a(cosmo, jnp.atleast_1d(a))
    fde = jc.background.f_de(cosmo, jnp.atleast_1d(a))
    D1f = f1 / a
    
    
    F1p = 1.5 * Omega_m_a * D1 / a**2 - (D1f / a) * (
      Omega_de_a - 0.5 * Omega_m_a + 2) / jc.background.growth_factor(cosmo, jnp.atleast_1d(1.))[0]
    
    
    epsilon = 1e-5
    dfde = (3 * cosmo.wa * (jnp.log(a - epsilon) - (a - 1) / (a - epsilon)) /
          jnp.power(jnp.log(a - epsilon), 2))
    
    dEa = 0.5 * (-3 * cosmo.Omega_m / a**4 -
                2 * cosmo.Omega_k / a**3 + dfde *
                cosmo.Omega_de * jnp.power(a, fde)) / E
    
    gf = (F1p * a**3 * E + D1f * a**3 * dEa +
          3 * a**2 * E * D1f)
    
    
    
    displacements = zeldovich(linear, box_size, 0.)
    psi = D1 * \
            jnp.c_[cic_interpolate(displacements[0], pos_lagrangian, box_size, n_bins), \
                   cic_interpolate(displacements[1], pos_lagrangian, box_size, n_bins), \
                   cic_interpolate(displacements[2], pos_lagrangian, box_size, n_bins)]
    del displacements
    
    state['momenta'] = a**2 * f1 * E * psi
    #state['forces' ] = a**2 * E * gf / D1 * psi
    state['forces' ] = a**2 * E * gf * psi
    state['positions'] = pos_lagrangian + psi
    state['positions'] += box_size
    state['positions'] %= box_size
    
    return state
    
    
    
    
@jax.jit
def nbody(cosmo, state, stages, rho, n_bins, box_size):
  
    
    ai = stages[0]

    # first force calculation for jump starting
    state['forces'] = force(cosmo, state['positions'], rho, box_size)
    

    x, p, f = ai, ai, ai
    
    xs = jnp.arange(stages.shape[0] - 1)
    init = (state, x, p, f, box_size, stages)
    def scan_func(carry, i):
        state, x, p, f, box_size, stages = carry
        
        a0 = stages[i]
        a1 = stages[i + 1]
        ah = (a0 * a1)**0.5
        state['momenta'] += kick(cosmo, state['forces'], p, f, ah)
        p = ah

      # Drift step
        state['positions'] += drift(cosmo, state['momenta'], x, p, a1)# + box_size
        #state['positions'] %= box_size
      # Optional PGD correction step
        
        x = a1

      # Force
        state['forces'] = force(cosmo, state['positions'], rho, box_size)
        f = a1

      # Kick again
        state['momenta'] += kick(cosmo, state['forces'], p, f, a1)
        p = a1
        
        return (state, x, p, f, box_size, stages), None
      
    (state, _, _, _, _, _), _ = jax.lax.scan(scan_func, init, xs)
    return state
'''
    # Loop through the stages
    for i in range(len(stages) - 1):
        a0 = stages[i]
        a1 = stages[i + 1]
        ah = (a0 * a1)**0.5
        print(state['positions'])
      # Kick step
        state['momenta'] += kick(cosmo, state['forces'], p, f, ah)
        p = ah

      # Drift step
        state['positions'] += drift(cosmo, state['momenta'], x, p, a1)
      # Optional PGD correction step
        
        x = a1

      # Force
        state['forces'] = force(cosmo, state['positions'], rho, box_size)
        f = a1

      # Kick again
        state['momenta'] += kick(cosmo, state['forces'], p, f, a1)
        p = a1
        

    
    return state
'''


def gen_ode_func(rho, mesh_shape):
    
    def nbody_ode(state, a, cosmo):
        """
        state is a tuple (position, velocities)
        """
        pos, vel = state

        forces = force(cosmo, pos, rho, mesh_shape[0]) * 1.5 * cosmo.Omega_m

        # Computes the update of position (drift)
        dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel
        
        # Computes the update of velocity (kick)
        dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces
        
        return dpos, dvel

    return nbody_ode