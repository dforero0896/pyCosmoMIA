import jax 
import jax.numpy as jnp
import sys
sys.path.append("/home/astro/dforero/codes/pyCosmoMIA/")
from cosmomia import cy_read_cic #, py_read_cic


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


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    field = np.random.uniform(size = (10, 10, 10)).astype(np.double)
    box_size = np.array([10.,10.,10.])
    box_min = np.array([0.,0.,0.])    
    positions = (box_size - box_min)[None,:] * np.random.uniform(size = (10, 3)) + box_min[None,:]
    #results_c = py_read_cic(field.ravel(), np.array(field.shape, dtype = np.int32), positions, box_size, box_min)
    results_py = interpolate_field(field, positions, box_min[0], box_min[1], box_min[2], field.shape[0], box_size[0])
    results_cy = np.array([cy_read_cic["double", "int"](field.ravel(), positions[i,:], box_size, box_min, np.array(field.shape, dtype = np.int32), 1) for i in range(positions.shape[0])])
    #print(results_c)
    print(results_py)
    print(results_cy)