# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
from libc.stdio cimport printf, fflush, stdout
from libc.stdlib cimport abort, malloc, free
from libc.math cimport floor, sqrt, round, abs, exp, acos, asin, cos, sin
from libcpp.vector cimport vector
from libcpp.iterator cimport iterator, front_insert_iterator, input_iterator_tag
from libcpp cimport bool as cbool
from cython cimport boundscheck, wraparound, numeric, floating, integral, cdivision, inline
import numpy as np
cimport numpy as cnp
import jax
from cython.parallel cimport parallel, prange
cimport openmp
import multiprocessing
#from scipy.spatial import KDTree
#import pybosque
from pykdtree.kdtree import KDTree
import time

#cdef extern from "<iterator>" namespace "std":
    ##cdef cppclass iterator[Category,T,Distance,Pointer,Reference]:
    #    pass
    #T make_move_iterator[T](T i) nogil
    #cdef cppclass input_iterator_tag

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() nogil# we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed)  nogil# not worrying about matching the exact int type for seed
    
    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution()  nogil
        uniform_real_distribution(T a, T b)  nogil
        T operator()(mt19937 gen)  nogil # ignore the possibility of using other classes for "gen"

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()  nogil
        uniform_int_distribution(T a, T b)  nogil
        T operator()(mt19937 gen)  nogil # ignore the possibility of using other classes for "gen"

    cdef cppclass normal_distribution[T]:
        normal_distribution()  nogil
        normal_distribution(T a, T b) nogil
        T operator()(mt19937 gen)   nogil# ignore the possibility of using other classes for "gen"
    cdef cppclass exponential_distribution[T]:
        exponential_distribution()  nogil
        exponential_distribution(T b) nogil
        T operator()(mt19937 gen)   nogil# ignore the possibility of using other classes for "gen"
    cdef cppclass discrete_distribution[T]:
        discrete_distribution()  nogil
        #discrete_distribution(iterator[input_iterator_tag, int, ptrdiff_t, int*, int&] first, iterator[input_iterator_tag, int, ptrdiff_t, int*, int&]  last)  nogil
        #InputIt discrete_distribution[InputIt](InputIt firstW, InputIt lastW)  nogil
        T operator()(mt19937 gen)   nogil# ignore the possibility of using other classes for "gen"
    cdef cppclass chi_squared_distribution[T]:
        chi_squared_distribution()  nogil
        chi_squared_distribution(T b) nogil
        T operator()(mt19937 gen)   nogil# ignore the possibility of using other classes for "gen"

# Declare external functions from the C++ file
cdef extern from "src/utils.h":
    cdef discrete_distribution[int] setup_discrete_distribution(vector[double] bin_counts) nogil


#cdef extern from "src/subgrid.h":
#    cdef cppclass SubgridCatalog:
#        vector[vector[double]] pos
#        vector[vector[double]] vel
#        vector[cbool] is_dm
#        vector[double] dweb
#        vector[double] delta_dm
#        vector[double] r_min
#        vector[double] delta_max
#        vector[cbool] attractor
#        float param1
#        float param2
#        size_t num_particles
#
#    int wrap_indices(int value, int size) nogil
#    
#    T read_cic[T](const vector[T]& field,
#                const vector[T]& position,
#                const vector[T]& box_size,
#                const vector[T]& box_min,
#                const vector[int]& dims,
#                cbool wrap) nogil
#
#    vector[cbool] is_attractor_fun(const vector[cbool]& is_dm, const vector[double]& cw_type)
#
#    SubgridCatalog assign_particles_to_gals(
#        const vector[vector[double]]& dm_particles, 
#        const vector[unsigned int]& target_ncount, 
#        int grid_x, int grid_y, int grid_z,
#        const vector[double]& box_size,
#        const vector[double]& box_min,
#        const vector[double]& dm_cw_type,
#        const vector[double]& dm_dens,
#        const vector[vector[double]]& displacement,
#        const vector[double]& velocities,
#        double dist,
#        cbool debug)
#


cdef Py_ssize_t INDEX(Py_ssize_t i,Py_ssize_t j,Py_ssize_t k, Py_ssize_t nx, Py_ssize_t ny, Py_ssize_t nz) noexcept nogil:
    return k + j*nz + i*ny*nz
cdef void copy_view_to_vector(Py_ssize_t size, vector[numeric]& vector, numeric[:] view) noexcept nogil:
    for i in range(size):
        vector.push_back(view[i])
    
cdef Py_ssize_t wrap_indices(Py_ssize_t value, Py_ssize_t size) noexcept nogil:
    if (value < 0): return size + value
    if (value >= size): return value - size
    return value;

#def py_read_cic(double[:] field, int[:] dims, double[:,:] positions, double[:] box_size, double[:] box_min):
#
#    cdef vector[int] cdims
#    cdef vector[double] cfield
#    cdef vector[double] position
#    cdef vector[double] cbox_size
#    cdef vector[double] cbox_min
#    cdef cbool wrap = 1
#    copy_view_to_vector[int](3, cdims, dims)
#    copy_view_to_vector[double](cdims[0] * cdims[1] * cdims[2], cfield, field)
#    copy_view_to_vector[double](positions.shape[1], cbox_size, box_size)
#    copy_view_to_vector[double](positions.shape[1], cbox_min, box_min)
#    results = np.zeros(positions.shape[0], dtype = np.double)
#    for i in range(positions.shape[0]):
#        position.clear()
#        copy_view_to_vector(positions.shape[1], position, positions[i,:])
#        results[i] = read_cic(cfield,
#                            position,
#                            cbox_size,
#                            cbox_min,
#                            cdims,
#                            wrap)
#    return results
#
cdef floating cy_weights(floating ddx, floating ddy, floating ddz, floating ii, floating jj, floating kk) noexcept nogil:
    return (((1 - ddx) + ii * (-1 + 2 * ddx)) * 
            ((1 - ddy) + jj * (-1 + 2 * ddy)) *
            ((1 - ddz) + kk * (-1 + 2 * ddz)))


@boundscheck(False)
@wraparound(False)
cpdef floating cy_read_cic(floating[:] field, 
                floating[:] position, 
                floating[:] box_size, 
                floating[:] box_min,
                Py_ssize_t[:] dims,
                cbool wrap)  noexcept nogil:
    
    #cdef vector[floating] cell_size = [box_size[0] / dims[0], box_size[1] / dims[1], box_size[2] / dims[2]]
    cdef vector[floating] cell_size 
    cdef Py_ssize_t a
    for a in range(3):
        cell_size.push_back(box_size[0] / dims[0])

    cdef floating xpos = (position[0] - box_min[0]) / cell_size[0]
    cdef floating ypos = (position[1] - box_min[1]) / cell_size[1]
    cdef floating zpos = (position[2] - box_min[2]) / cell_size[2]

    cdef Py_ssize_t i = <Py_ssize_t>(floor(xpos));
    cdef Py_ssize_t j = <Py_ssize_t>(floor(ypos));
    cdef Py_ssize_t k = <Py_ssize_t>(floor(zpos));

    cdef floating ddx = xpos - i;
    cdef floating ddy = ypos - j;
    cdef floating ddz = zpos - k;
    cdef floating result = 0;
    cdef Py_ssize_t ii, jj, kk, index_3d
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                index_3d = INDEX(wrap_indices(i+ii, dims[0]), 
                                wrap_indices(j+jj, dims[1]), 
                                wrap_indices(k+kk, dims[2]),
                                dims[0],
                                dims[1],
                                dims[2]);
                result += cy_weights[floating](ddx, ddy, ddz, ii, jj, kk) * field[index_3d];
    return result;


@boundscheck(False)
@wraparound(False)
cpdef floating cy_read_cic_floats(floating[:] field, 
                floating position_x, floating position_y, floating position_z, 
                floating box_size_x, floating box_size_y, floating box_size_z, 
                floating box_min_x, floating box_min_y, floating box_min_z,
                Py_ssize_t dims_x, Py_ssize_t dims_y, Py_ssize_t dims_z,
                cbool wrap)  noexcept nogil:
    
    #cdef vector[floating] cell_size = [box_size[0] / dims[0], box_size[1] / dims[1], box_size[2] / dims[2]]
    cdef floating cell_size_x, cell_size_y, cell_size_z
    cdef Py_ssize_t a
    
    cell_size_x = box_size_x / dims_x
    cell_size_y = box_size_y / dims_y
    cell_size_z = box_size_z / dims_z

    cdef floating xpos = (position_x - box_min_x) / cell_size_x
    cdef floating ypos = (position_y - box_min_y) / cell_size_y
    cdef floating zpos = (position_z - box_min_z) / cell_size_z

    cdef Py_ssize_t i = <Py_ssize_t>(floor(xpos));
    cdef Py_ssize_t j = <Py_ssize_t>(floor(ypos));
    cdef Py_ssize_t k = <Py_ssize_t>(floor(zpos));

    cdef floating ddx = xpos - i;
    cdef floating ddy = ypos - j;
    cdef floating ddz = zpos - k;
    cdef floating result = 0;
    cdef Py_ssize_t ii, jj, kk, index_3d
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                index_3d = INDEX(wrap_indices(i+ii, dims_x),
                                wrap_indices(j+jj, dims_y),
                                wrap_indices(k+kk, dims_z),
                                dims_x,
                                dims_y,
                                dims_z);
                result += cy_weights[floating](ddx, ddy, ddz, ii, jj, kk) * field[index_3d];
    return result;

@boundscheck(False)
@wraparound(False)
cpdef void cy_read_cic_vector(floating[:] output,
                                  floating[:] field,
                                  floating[:,:] positions,
                                  floating[:] box_size,
                                  floating[:] box_min,
                                  Py_ssize_t[:] dims,
                                  cbool wrap,
                                  size_t n_threads,
                                    ) noexcept nogil:

    cdef Py_ssize_t i, size
    size = positions.shape[0]
    if size != output.shape[0]:
        printf("ERROR: Size of output array does not match size of particles array.\n")
        fflush(stdout)
        abort()
    for i in prange(size, nogil = True, num_threads = n_threads):
        output[i] += cy_read_cic_floats[floating](field, 
                                                 positions[i,0], positions[i,1], positions[i,2],
                                                 box_size[0], box_size[1], box_size[2],
                                                 box_min[0], box_min[1], box_min[2],
                                                 dims[0], dims[1], dims[2],
                                                 wrap)


cpdef void cy_cic_mas(floating[:,:,:] field, 
                      floating[:] x, floating[:] y, floating[:] z,
                      floating[:] w,
                      floating[:] box_min,
                      floating[:] box_size,
                      cbool wrap,
                      size_t n_threads,
                    ) noexcept nogil:
    
    cdef vector[Py_ssize_t] n_bins;
    cdef vector[floating] bin_size
    cdef size_t i 
    for i in range(3):
        n_bins.push_back(field.shape[i])
        bin_size.push_back(box_size[i] / field.shape[i])

    


    cdef floating x_, y_, z_, wx0, wx1, wy0, wy1, wz0, wz1
    cdef Py_ssize_t x0, y0, z0, x1, y1, z1


    

    for i in prange(x.shape[0], nogil = True, num_threads = n_threads):
    #for i in range(x.shape[0]):
        x_ = (x[i] - box_min[0]) / bin_size[0]
        y_ = (y[i] - box_min[1]) / bin_size[1]
        z_ = (z[i] - box_min[2]) / bin_size[2]

        x0 = <size_t> floor(x_)
        y0 = <size_t> floor(y_)
        z0 = <size_t> floor(z_)


        wx1 = x_ - x0
        wx0 = 1 - wx1
        wy1 = y_ - y0
        wy0 = 1 - wy1
        wz1 = z_ - z0
        wz0 = 1 - wz1

        if wrap:
            x0 = wrap_indices(x0, n_bins[0])
            y0 = wrap_indices(y0, n_bins[1])
            z0 = wrap_indices(z0, n_bins[2])

        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        if wrap:
            x1 = wrap_indices(x1, n_bins[0])
            y1 = wrap_indices(y1, n_bins[1])
            z1 = wrap_indices(z1, n_bins[2])

        wx0 = wx0 * w[i]
        wx1 = wx1 * w[i]


        field[x0,y0,z0] += wx0 * wy0 * wz0
        field[x1,y0,z0] += wx1 * wy0 * wz0
        field[x0,y1,z0] += wx0 * wy1 * wz0
        field[x0,y0,z1] += wx0 * wy0 * wz1
        field[x1,y1,z0] += wx1 * wy1 * wz0
        field[x1,y0,z1] += wx1 * wy0 * wz1
        field[x0,y1,z1] += wx0 * wy1 * wz1
        field[x1,y1,z1] += wx1 * wy1 * wz1




cdef int is_double_prec(floating a):
    if floating is double:
        return 1
    elif floating is float:
        return 0
cdef cbool check_in_range(floating[:] pos, floating[:] grid_center, floating[:] bin_size)  noexcept nogil:
    cdef cbool ret = 1
    cdef Py_ssize_t i = 0
    cdef floating min_, max_
    for i in range(3):
        min_ = (grid_center[i] - 0.5 * bin_size[i])
        max_ = (grid_center[i] + 0.5 * bin_size[i])
        ret = ret and (pos[i] >= min_) and (pos[i] <= max_)
        if not ret:
            printf("axis %i pos_i %lf min_i %lf max_i %lf\n", i, pos[i], min_, max_)
            fflush(stdout)
    return ret


def print_icon():
    print("""                                                                                                                              
                                                                             @@@                                              
                                            @@ @@@@@ @@@@@@      @@@@@@     @@@@       @@                                     
                                    @@@@    @@@   @@@    @@@   @@@   @@@    @@@@      @@@    @                                
                                  @@   @    @@@    @@    @@@  @@@     @@@   @@@@    @@@@@     @@@@@                           
                 @       @@@@@@   @@@       @@@    @@    @@@  @@@     @@@  @@@@@   @@@@@     @@@@                             
             @@@@@@     @@    @@@  @@@@@@@  @@@    @@@    @@  @@@    @@@@  @  @@@ @@ @@@     @@@          @                   
           @@      @   @@@     @@@     @@@@ @@@    @@@   @@@@@ @@@   @@   @@  @@@@  @@@@    @@@         @@@                   
          @@@          @@@     @@@ @@    @@@@@@@@ @@             @@@@   @@@@@ @@@   @@@    @@@       @@@@@@                   
          @@@         @ @@@     @@  @@@@@                                     @@  @@@@@   @@@@     @@@  @@@                   
          @@@@         @ @@@   @@                                                     @@@@@@@    @@ @@ @@@                    
           @@@@        @@                                                                  @@@ @@@     @@@                    
             @@@@     @@                                                                       @@@     @@@                    
              @@@@@@@@                                                                                @@@@                    
                                                                                                      @@@                                         
                                          @@@@@@@@@@@@@@@      @@@@@@@@@@@@@@@                                                
                                     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                           
                                   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                        
                                 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                      
                               @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@                                    
                             @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@@@@@@@@@@@@@                                   
                            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@       @@@@@@@@@@@@@@@                                 
                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@   @@@@@@@@@@@@@@@@                                
                         @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                               
                        @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                              
                       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                             
                       @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                            
                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                            
                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                           
                      @@@@@@@@@@@@@@           @@@@@@@@@@@@@@@@@@@@@@@@@@           @@@@@@@@@@@@@@@                           
                      @@@@@@@@@@@@@              @@@@@@@@@@@@@@@@@@@@@@              @@@@@@@@@@@@@@                           
                      @@@@@@@@@@@@@@@@             @@@@@@@@@@@@@@@@@@@             @@@@@@@@@@@@@@@@                           
                      @@@@@@@@@@@@@@@@@@           @@@@@@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@@@@                            
                      @@@@@@@@@@@@@@@@@@           @@@@@@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@@@@                            
                       @@@@@@@@@@@@@@@@@           @@@@@@@@@@@@@@@@@@@          @@@@@@@@@@@@@@@@@@                            
                        @@@@@@@@@@@@@@@           @@@@@@@@@@@@@@@@@@@@@           @@@@@@@@@@@@@@                              
                          @@@@@@@@@@             @@@@@@@@@@@@@@@@@@@@@@              @@@@@@@@@                                
                         @@                    @@@@@@@@@@@@@@@@@@@@@@@@@@@                   @@@                              
                         @@@@@@@          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@         @@@@@@@                               
                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                               
                           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                
                            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                  
                              @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                   
                                @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                      
                                  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                        
                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                           
                                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                
                                                @@@@@@@@@@@@@@@@@@@@@@@@                                                      
                                                                                                                              """)

#@boundscheck(False)
#@wraparound(False)
cpdef dict py_assign_particles_to_gals(floating[:,:] dm_particles, unsigned int[:] target_ncount,
                                Py_ssize_t grid_size, floating[:] box_size, floating[:] box_min,
                                unsigned int[:] dm_cw_type, floating[:] dm_dens, floating[:,:] displacement,
                                floating[:,:] velocities, size_t seed, floating dist_std_par, 
                                size_t small_scale_dist,
                                size_t fully_rand_dist,
                                cbool debug = False ):
    

    print_icon()
    cdef int is_double = is_double_prec(dm_particles[0,0])
    if is_double:
        dtype = np.double
    else:
        dtype = np.float32
    
    cdef unsigned int[:] number_dm
    if debug:
        number_dm = np.zeros(target_ncount.shape[0], dtype = np.uintc)
    
    printf("Creating particle catalog\n")
    fflush(stdout)

    cdef floating[:] bin_size = np.zeros_like(box_size)
    cdef Py_ssize_t i, ii, jj, kk, index_3d, par, number_cen_in_cell, cen_idx, iii
    for i in range(3):
        bin_size[i] = box_size[i] / grid_size


    cdef mt19937 gen = mt19937(seed)
    cdef uniform_real_distribution[double] dist_unif = uniform_real_distribution[double](0.0,1.0)
    cdef uniform_int_distribution[int] dist_int
    cdef normal_distribution[floating] dist_gauss = normal_distribution[floating](0., dist_std_par)
    cdef exponential_distribution[floating] dist_exp = exponential_distribution[floating](1. / dist_std_par)
    cdef chi_squared_distribution[floating] dist_chi = chi_squared_distribution[floating](dist_std_par)
 

    cdef floating[:] grid_center = np.zeros(3, dtype = dtype)
    #cdef Py_ssize_t[:] grid_indices = np.zeros(3, dtype = np.int_)
    cdef Py_ssize_t grid_indices[3]


    printf("Assigning DM particles to cells...")
    fflush(stdout)
    cdef vector[vector[Py_ssize_t]] dm_per_cell
    dm_per_cell.reserve(grid_size**3)
    dm_per_cell.resize(grid_size**3)
    #for i in range(grid_size**3):
    #    #dm_per_cell.push_back([])
    #    dm_per_cell.push_back(*(new vector[Py_ssize_t]()))
    for i in range(dm_particles.shape[0]):
        ii = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,0] / box_size[0])), grid_size)
        jj = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,1] / box_size[1])), grid_size)
        kk = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,2] / box_size[2])), grid_size)
        index_3d = INDEX(ii, jj, kk, grid_size, grid_size, grid_size)
        dm_per_cell[index_3d].push_back(i)
        if debug:

            grid_indices[2] = <Py_ssize_t> (index_3d % grid_size)
            grid_indices[1] = <Py_ssize_t> (index_3d / grid_size) % grid_size
            grid_indices[0] = <Py_ssize_t> (index_3d / (grid_size * grid_size))
        
            for iii in range(3):
                grid_center[iii] = (grid_indices[iii] + 0.5) * bin_size[iii]

            number_dm[index_3d] += 1
            if not check_in_range[floating](dm_particles[i, :], grid_center, bin_size):
                printf("DM Particle %li not in cell", i)
                printf("%li %li %li", grid_indices[0], grid_indices[1], grid_indices[2])
                printf("%li %li %li", ii,jj,kk)
                fflush(stdout)
                abort()
            
    printf(" Done\n")
    #cdef Py_ssize_t total_target_tracers = np.sum(target_ncount)
    cdef Py_ssize_t total_target_tracers = 0
    for i in range(target_ncount.shape[0]):
        total_target_tracers += target_ncount[i]
    printf("%li particles are required.\n", total_target_tracers)
    printf("Allocating memory for results\n")
    pos = np.empty((total_target_tracers, 3), dtype = dtype)
    cdef floating [:, :] pos_view = pos
    vel = np.empty((total_target_tracers, 3), dtype = dtype)
    cdef floating [:, :] vel_view = vel
    is_dm = np.empty((total_target_tracers,), dtype = np.int32)
    cdef int [:] is_dm_view = is_dm
    is_attractor = np.empty((total_target_tracers,), dtype = np.int32)
    cdef int [:] is_attractor_view = is_attractor
    dweb = np.empty((total_target_tracers,), dtype = np.int16)
    cdef short [:] dweb_view = dweb
    delta_dm = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] delta_dm_view = delta_dm
    r_min = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] r_min_view = r_min
    delta_max = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] delta_max_view = delta_max

    cdef size_t used_dm = 0
    cdef size_t sampled_around = 0
    cdef size_t sampled_rand = 0

    
    cdef floating[:] particle_tmp = np.zeros(3, dtype = dtype)
    cdef Py_ssize_t[:] dims = np.array([grid_size, grid_size, grid_size])

    cdef vector[Py_ssize_t] dm_particles_in_cell
    cdef Py_ssize_t number_dm_in_cell
    cdef floating draw
    cdef vector[floating] psi_i
    psi_i.resize(3)
    cdef int dummy = 0
    cdef int missing_counter = 0
    
    cdef Py_ssize_t particle_counter = 0
    cdef floating displacement_draw 
    cdef int hist_bin_id
    
    printf("Assigning particles to tracers...\n")
    for index_3d in range(grid_size**3):
        assigned_random_flag = 0
        number_cen_in_cell = target_ncount[index_3d]
        if number_cen_in_cell == 0: continue

        grid_indices[2] = <Py_ssize_t> (index_3d % grid_size)
        grid_indices[1] = <Py_ssize_t> (index_3d / grid_size) % grid_size
        grid_indices[0] = <Py_ssize_t> (index_3d / (grid_size * grid_size))
        
        for ii in range(3):
            grid_center[ii] = (grid_indices[ii] + 0.5) * bin_size[ii]
        if debug:
            dummy = INDEX(grid_indices[0], grid_indices[1], grid_indices[2], grid_size, grid_size, grid_size)
            
            if index_3d != dummy:
                printf("Error in index manipulations. %li %li %li %li %li\n", index_3d, grid_indices[0], grid_indices[1], grid_indices[2], dummy)
                abort()
        dm_particles_in_cell = dm_per_cell[index_3d]
        number_dm_in_cell = dm_particles_in_cell.size()
        if debug:
            for i in range(number_dm_in_cell):
                if not check_in_range[floating](dm_particles[dm_particles_in_cell[i], :], grid_center, bin_size):
                    print(f"DM Particle {dm_particles_in_cell[i]} not in cell {index_3d}")
                    print(grid_indices[0], grid_indices[1], grid_indices[2])
                    abort()
        #print(number_dm_in_cell)
        if debug:
            if number_dm_in_cell != number_dm[index_3d]:
                printf("Something wrong assigning dm particles to cells %li != %li\n", number_dm_in_cell, number_dm[index_3d])
                abort()


        for par in range(number_cen_in_cell):
            #if False:#par < number_dm_in_cell:
            if par < number_dm_in_cell:
                for ii in range(3):
                    pos_view[particle_counter, ii] = dm_particles[dm_particles_in_cell[par], ii]
                    if debug:
                        if not check_in_range[floating](dm_particles[dm_particles_in_cell[par], :], grid_center, bin_size):
                            printf("DM Particle %li not in cell\n", dm_particles_in_cell[par])
                            abort()
                is_dm_view[particle_counter] = 1
                used_dm += 1
            
            else:
                missing_counter = par - number_dm_in_cell
                #if False:#number_dm_in_cell > 0:
                if number_dm_in_cell > 0:
                    dist_int = uniform_int_distribution[int](0,number_dm_in_cell-1)
                    
                    #missing_counter = dist_int(gen) if missing_counter > number_dm_in_cell else  missing_counter
                    missing_counter = dist_int(gen)

                    if small_scale_dist == 1:
                        displacement_draw = dist_gauss(gen) 
                    elif small_scale_dist == 2:
                        displacement_draw  = (dist_exp(gen))
                    elif small_scale_dist == 5:
                        displacement_draw = dist_chi(gen)
                    else:
                        printf("ERROR: Received `small_scale_dist = %i`, accepted values are 1 = gaussian, 2 = exponential)\n", small_scale_dist)
                        fflush(stdout)
                        abort()
                    for ii in range(3):
                        pos_view[particle_counter, ii] = dm_particles[dm_particles_in_cell[missing_counter], ii] if missing_counter < number_dm_in_cell  else grid_center[ii]
                        if debug:
                            if not check_in_range[floating](dm_particles[dm_particles_in_cell[missing_counter], :], grid_center, bin_size):
                                printf("DM Particle %li not in cell either\n", dm_particles_in_cell[missing_counter])
                                abort()
                        pos_view[particle_counter, ii] -= displacement[dm_particles_in_cell[missing_counter], ii]
                        #displacement_draw = clip(displacement_draw, -sqrt(3) * bin_size[ii], sqrt(3) * bin_size[ii])
                        pos_view[particle_counter, ii] += displacement_draw
                        
                        
                    for ii in range(3):
                        #psi_i[ii] = cy_read_cic[floating](displacement[:,ii],
                        #                                                            pos_view[particle_counter,:],
                        #                                                            box_size,
                        #                                                            box_min,
                        #                                                            dims,
                        #                                                            1)
                        psi_i[ii] = displacement[dm_particles_in_cell[missing_counter], ii]
                    for ii in range(3):
                        pos_view[particle_counter, ii] += psi_i[ii]
                        pos_view[particle_counter, ii] = (pos_view[particle_counter, ii] + box_size[ii]) % box_size[ii]
                    sampled_around += 1
                elif number_dm_in_cell == 0 and assigned_random_flag:
                    if small_scale_dist == 1:
                        displacement_draw = dist_gauss(gen) 
                    elif small_scale_dist == 2:
                        displacement_draw = (dist_exp(gen))
                    elif small_scale_dist == 5:
                        displacement_draw = dist_chi(gen)
                    else:
                        printf("ERROR: Received `small_scale_dist = %i`, accepted values are 1 = gaussian, 2 = exponential)\n", small_scale_dist)
                        fflush(stdout)
                        abort()
                    
                    for ii in range(3):
                        #displacement_draw = clip(displacement_draw, -sqrt(3) * bin_size[ii], sqrt(3) * bin_size[ii])
                        pos_view[particle_counter, ii] = pos_view[particle_counter-1, ii] + displacement_draw
                    sampled_around += 1
                else: #Need a random particle, no DM in cell

                    #for ii in range(3):
                    #    pos_view[particle_counter,ii] = grid_center[ii]
                    #move_centers(1e-1, grid_indices[0], grid_indices[1], grid_indices[2], pos_view[particle_counter,:], dm_dens, grid_size, box_size)
                    for ii in range(3):
                        if fully_rand_dist == 3:
                            draw = 2 * dist_unif(gen) - 1
                            #pos_view[particle_counter, ii] = ((grid_center[ii] + (0.5 * bin_size[ii] * draw)) + box_size[ii]) % box_size[ii]
                            if draw >= 0:
                                pos_view[particle_counter, ii] = (grid_center[ii] + (0.5 * bin_size[ii] * (1 - sqrt(draw))) + box_size[ii]) % box_size[ii]
                            else:
                                pos_view[particle_counter, ii] = (grid_center[ii] + (0.5 * bin_size[ii] * (-1) * (1 - sqrt(-draw))) + box_size[ii]) % box_size[ii]
                        elif fully_rand_dist == 1:
                            draw = dist_gauss(gen)
                            pos_view[particle_counter, ii] = (grid_center[ii] + draw  + box_size[ii]) % box_size[ii]
                        elif fully_rand_dist == 2:
                            draw = dist_exp(gen)
                            pos_view[particle_counter, ii] = (grid_center[ii] + draw  + box_size[ii]) % box_size[ii]
                        else:
                            printf("ERROR: Received `fully_rand_dist = %i`, accepted values are 1 = gaussian, 2 = exponential, 3 = triangular)\n", fully_rand_dist)
                            fflush(stdout)
                            abort()
                    sampled_rand += 1
                    assigned_random_flag = 1
                    
                is_dm_view[particle_counter] = 0
                
            dweb_view[particle_counter] = dm_cw_type[index_3d]
            
            delta_dm_view[particle_counter] =  cy_read_cic[floating](dm_dens,
                                                    pos_view[particle_counter,:],
                                                    box_size,
                                                    box_min,
                                                    dims,
                                                    1)
            for ii in range(3):
                vel_view[particle_counter,ii] =  cy_read_cic[floating](velocities[:,ii],
                                                    pos_view[particle_counter,:],
                                                    box_size,
                                                    box_min,
                                                    dims,
                                                    1)
            r_min_view[particle_counter] = 1e10
            delta_max_view[particle_counter] = -1e10
            is_attractor_view[particle_counter] = (dweb_view[particle_counter] < 4) and is_dm_view[particle_counter]
            #if not is_attractor_view[particle_counter]:
            #    printf("Found not attractor: %i %i\n", (dweb_view[particle_counter] < 4), is_dm_view[particle_counter])
            #    fflush(stdout)
            #    abort()
            #if debug:
            #    if not check_in_range[floating](pos_view[particle_counter,:], grid_center, bin_size):
            #        print(f"Particle {particle_counter} not in cell")
            #        #abort()
            particle_counter+=1
    print(f"Used {used_dm} DM particles")
    print(f"Created {sampled_around} particles around existig DM")
    print(f"Created {sampled_rand} particles randomly in empty cells")
    print(f"For a total of {total_target_tracers} == {particle_counter} == {used_dm + sampled_around + sampled_rand} particles")
    print(f"Sampled {100 * sampled_around / (used_dm + sampled_around + sampled_rand)} % of particles around DM.")
    print(f"Sampled {100 * sampled_rand / (used_dm + sampled_around + sampled_rand)} % of particles in empty cells.")
    print("Done")

    return dict(pos = pos, vel = vel, is_dm = is_dm, dweb = dweb, delta_dm = delta_dm, r_min = r_min, delta_max = delta_max, is_attractor = is_attractor, order = np.arange(pos.shape[0]))
    
            


cpdef dict par_py_assign_particles_to_gals(floating[:,:] dm_particles, unsigned int[:] target_ncount,
                                Py_ssize_t grid_size, floating[:] box_size, floating[:] box_min,
                                unsigned int[:] dm_cw_type, floating[:] dm_dens, floating[:,:] displacement,
                                floating[:,:] velocities, size_t seed, floating dist_std_par, cbool debug):

    printf("\nERROR: Parallel version of function is outdated, please use the serial version.\n")
    fflush(stdout)
    abort()
    print_icon()
    cdef int is_double = is_double_prec(dm_particles[0,0])
    if is_double:
        dtype = np.double
    else:
        dtype = np.float32
    
    cdef unsigned int[:] number_dm
    if debug:
        number_dm = np.zeros(target_ncount.shape[0], dtype = np.uintc)
    
    printf("Creating particle catalog\n")
    fflush(stdout)

    cdef floating[:] bin_size = np.zeros_like(box_size)
    cdef Py_ssize_t i, ii, jj, kk, index_3d, par, cen_idx, iii, num_threads, thread_id, j
    for i in range(3):
        bin_size[i] = box_size[i] / grid_size


    num_threads = multiprocessing.cpu_count()
    printf("MP Using %i threads\n", num_threads)
    fflush(stdout)

    cdef mt19937 gen = mt19937(seed)
    cdef uniform_real_distribution[double] dist_unif = uniform_real_distribution[double](0.0,1.0)
    cdef uniform_int_distribution[int] dist_int
    cdef normal_distribution[floating] dist_gauss = normal_distribution[floating](0., dist_std_par)

    
    cdef vector[vector[floating]] grid_center
    grid_center.resize(num_threads)
    cdef vector[vector[Py_ssize_t]] grid_indices
    grid_indices.resize(num_threads)
    for i in range(num_threads):
        grid_indices[i].resize(3)
        grid_center[i].resize(3)
    
    #cdef Py_ssize_t[:] grid_indices = np.zeros(3, dtype = np.int_)
    


    printf("Assigning DM particles to cells...")
    fflush(stdout)
    cdef vector[vector[Py_ssize_t]] dm_per_cell
    dm_per_cell.reserve(grid_size**3)
    dm_per_cell.resize(grid_size**3)
    #for i in range(grid_size**3):
    #    #dm_per_cell.push_back([])
    #    dm_per_cell.push_back(*(new vector[Py_ssize_t]()))
    for i in range(dm_particles.shape[0]):
        ii = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,0] / box_size[0])), grid_size)
        jj = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,1] / box_size[1])), grid_size)
        kk = wrap_indices(<Py_ssize_t>(floor(grid_size * dm_particles[i,2] / box_size[2])), grid_size)
        index_3d = INDEX(ii, jj, kk, grid_size, grid_size, grid_size)
        dm_per_cell[index_3d].push_back(i)
   
            
            
    printf(" Done\n")
    #cdef Py_ssize_t total_target_tracers = np.sum(target_ncount)
    cdef Py_ssize_t total_target_tracers = 0
    for i in range(target_ncount.shape[0]):
        total_target_tracers += target_ncount[i]
    printf("%li particles are required.\n", total_target_tracers)
    
    cdef vector[vector[vector[floating]]] pos, vel
    cdef vector[vector[size_t]] is_dm, is_attractor, dweb
    cdef vector[vector[floating]] delta_dm, r_min, delta_max
    cdef vector[vector[floating]] pos_buffer, vel_buffer
    cdef vector[vector[Py_ssize_t]] dm_particles_in_cell
    pos.resize(num_threads)
    vel.resize(num_threads)
    is_dm.resize(num_threads)
    is_attractor.resize(num_threads)
    dweb.resize(num_threads)
    delta_dm.resize(num_threads)
    r_min.resize(num_threads)
    delta_max.resize(num_threads)
    pos_buffer.resize(num_threads)
    vel_buffer.resize(num_threads)
    dm_particles_in_cell.resize(num_threads)
    for i in range(num_threads):
        pos_buffer[i].resize(3)
        vel_buffer[i].resize(3)

    cdef vector[size_t] used_dm
    cdef vector[size_t] sampled_new
    cdef vector[int] missing_counter
    for i in range(num_threads):
        used_dm.push_back(0)
        sampled_new.push_back(0)
        missing_counter.push_back(0)

    
    
    cdef vector[Py_ssize_t] dims
    for i in range(3):
        dims.push_back(grid_size)

    
    cdef vector[Py_ssize_t] number_dm_in_cell, number_cen_in_cell
    number_dm_in_cell.resize(num_threads)
    number_cen_in_cell.resize(num_threads)
    cdef vector[floating] draw, psi_i
    draw.resize(num_threads)
    psi_i.resize(num_threads)
    cdef vector[int] dummy
    dummy.resize(num_threads)
    
    #z = Math.round(i / (WIDTH * HEIGHT));
    #y = Math.round((i - z * WIDTH * HEIGHT) / WIDTH);
    #x = i - WIDTH * (y + HEIGHT * z);
    
    printf("Assigning particles to tracers...\n")
    #with nogil, parallel():
    openmp.omp_set_dynamic(0)
    for index_3d in prange(grid_size**3, nogil=True):
        thread_id = openmp.omp_get_thread_num()
        number_cen_in_cell[thread_id] = target_ncount[index_3d]
        if number_cen_in_cell[thread_id] == 0: continue

        grid_indices[thread_id][2] = <Py_ssize_t> (index_3d % grid_size)
        grid_indices[thread_id][1] = <Py_ssize_t> (index_3d / grid_size) % grid_size
        grid_indices[thread_id][0] = <Py_ssize_t> (index_3d / (grid_size * grid_size))
        
        for ii in range(3):
            grid_center[thread_id][ii] = (grid_indices[thread_id][ii] + 0.5) * bin_size[ii]
        if debug:
            dummy[thread_id] = INDEX(grid_indices[thread_id][0], grid_indices[thread_id][1], grid_indices[thread_id][2], grid_size, grid_size, grid_size)
            
            if index_3d != dummy[thread_id]:
                printf("Error in index manipulations. %li %li %li %li %li\n", index_3d, grid_indices[thread_id][0], grid_indices[thread_id][1], grid_indices[thread_id][2], dummy[thread_id])
                abort()
        dm_particles_in_cell[thread_id] = dm_per_cell[index_3d]
        number_dm_in_cell[thread_id] = dm_particles_in_cell[thread_id].size()
    
        #print(number_dm_in_cell)
        if debug:
            if number_dm_in_cell[thread_id] != number_dm[index_3d]:
                printf("Something wrong assigning dm particles to cells %li != %li\n", number_dm_in_cell[thread_id], number_dm[index_3d])
                abort()

        
        
        for par in range(number_cen_in_cell[thread_id]):
            #if False:#par < number_dm_in_cell:
            if par < number_dm_in_cell[thread_id]:
                for ii in range(3):
                    pos_buffer[thread_id][ii] = dm_particles[dm_particles_in_cell[thread_id][par], ii]
                    
                is_dm[thread_id].push_back(1)
                used_dm[thread_id] += 1
            else:
                missing_counter[thread_id] = par - number_dm_in_cell[thread_id]
                #if False:#number_dm_in_cell > 0:
                if number_dm_in_cell[thread_id] > 0:
                    dist_int = uniform_int_distribution[int](0,number_dm_in_cell[thread_id]-1)
                    #missing_counter = dist_int(gen) if missing_counter > number_dm_in_cell else  missing_counter
                    missing_counter[thread_id] = dist_int(gen)
                    
                    for ii in range(3):
                        pos_buffer[thread_id][ii] = dm_particles[dm_particles_in_cell[thread_id][missing_counter[thread_id]], ii] if missing_counter[thread_id] < number_dm_in_cell[thread_id]  else grid_center[thread_id][ii]
                        pos_buffer[thread_id][ii] -= displacement[dm_particles_in_cell[thread_id][missing_counter[thread_id]], ii]
                        pos_buffer[thread_id][ii] += dist_gauss(gen) #+ displacement[dm_particles_in_cell[cen_idx], ii]
                    for ii in range(3):
                        #psi_i = cy_read_cic[floating](displacement[:,ii],
                        #                                                            pos_view[particle_counter,:],
                        #                                                            box_size,
                        #                                                            box_min,
                        #                                                            dims,
                        #                                                            1)
                        psi_i[thread_id] = displacement[dm_particles_in_cell[thread_id][missing_counter[thread_id]], ii]
                        pos_buffer[thread_id][ii] += psi_i[thread_id]
                        pos_buffer[thread_id][ii] = (pos_buffer[thread_id][ii] + box_size[ii]) % box_size[ii]
                else:
                    for ii in range(3):
                        draw[thread_id] = 2 * dist_unif(gen) - 1
                        #pos_view[particle_counter, ii] = ((grid_center[ii] + (0.5 * bin_size[ii] * draw)) + box_size[ii]) % box_size[ii]
                        if draw[thread_id] >= 0:
                            pos_buffer[thread_id][ii] = (grid_center[thread_id][ii] + (0.5 * bin_size[ii] * (1 - sqrt(draw[thread_id]))) + box_size[ii]) % box_size[ii]
                        else:
                            pos_buffer[thread_id][ii] = (grid_center[thread_id][ii] + (0.5 * bin_size[ii] * (-1) * (1 - sqrt(-draw[thread_id]))) + box_size[ii]) % box_size[ii]
                is_dm[thread_id].push_back(0)
                sampled_new[thread_id] += 1
            pos[thread_id].push_back(pos_buffer[thread_id]) #maybe bug
            dweb[thread_id].push_back(dm_cw_type[index_3d])
            delta_dm[thread_id].push_back(cy_read_cic_floats[floating](dm_dens,
                                                    pos_buffer[thread_id][0], pos_buffer[thread_id][1], pos_buffer[thread_id][2],
                                                    box_size[0], box_size[1], box_size[2],
                                                    box_min[0], box_min[1], box_min[2],
                                                    dims[0], dims[1], dims[2],
                                                    1))
            for ii in range(3):
                vel_buffer[thread_id][ii] =  cy_read_cic_floats[floating](velocities[:,0],
                                                    pos_buffer[thread_id][0], pos_buffer[thread_id][1], pos_buffer[thread_id][2],
                                                    box_size[0], box_size[1], box_size[2],
                                                    box_min[0], box_min[1], box_min[2],
                                                    dims[0], dims[1], dims[2],
                                                    1)
            r_min[thread_id].push_back(1e10)
            delta_max[thread_id].push_back(-1e10)
            is_attractor[thread_id].push_back((dm_cw_type[index_3d] < 4) and is_dm[thread_id][is_dm[thread_id].size()-1])
            vel[thread_id].push_back(vel_buffer[thread_id])
            #if debug:
            #    if not check_in_range[floating](pos_view[particle_counter,:], grid_center, bin_size):
            #        print(f"Particle {particle_counter} not in cell")
            #        #abort()

    

    for i in range(1, num_threads):
        #pos[0].insert(
        #            pos[0].end(),
        #            make_move_iterator(pos[i].begin()),
        #            make_move_iterator(pos[i].end())
        #            )


        pos[0].insert(pos[0].end(), pos[i].begin(), pos[i].end())
        vel[0].insert(vel[0].end(), vel[i].begin(), vel[i].end())
        is_dm[0].insert(is_dm[0].end(), is_dm[i].begin(), is_dm[i].end())
        dweb[0].insert(dweb[0].end(), dweb[i].begin(), dweb[i].end())
        delta_dm[0].insert(delta_dm[0].end(), delta_dm[i].begin(), delta_dm[i].end())
        r_min[0].insert(r_min[0].end(), r_min[i].begin(), r_min[i].end())
        delta_max[0].insert(delta_max[0].end(), delta_max[i].begin(), delta_max[i].end())
        is_attractor[0].insert(is_attractor[0].end(), is_attractor[i].begin(), is_attractor[i].end())

        used_dm[0] += used_dm[i]
        sampled_new[0] += sampled_new[i]
        

    print(f"Used {used_dm[0]} DM particles")
    print(f"Created {sampled_new[0]} particles")
    print(f"For a total of {total_target_tracers} == {pos[0].size()} == {used_dm[0] + sampled_new[0]} particles")
    print(f"Sampled {100 * sampled_new[0] / (used_dm[0] + sampled_new[0])} % of particles.")
        #for j in range(pos[i].size()):
        #    pos[0].push_back(pos[i][j])

    printf("Allocating memory for results\n")
    _pos = np.empty((total_target_tracers, 3), dtype = dtype)
    cdef floating [:, :] pos_view = _pos
    _vel = np.empty((total_target_tracers, 3), dtype = dtype)
    cdef floating [:, :] vel_view = _vel
    _is_dm = np.empty((total_target_tracers,), dtype = np.int32)
    cdef int [:] is_dm_view = _is_dm
    _is_attractor = np.empty((total_target_tracers,), dtype = np.int32)
    cdef int [:] is_attractor_view = _is_attractor
    _dweb = np.empty((total_target_tracers,), dtype = np.int16)
    cdef short [:] dweb_view = _dweb
    _delta_dm = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] delta_dm_view = _delta_dm
    _r_min = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] r_min_view = _r_min
    _delta_max = np.empty((total_target_tracers,), dtype = dtype)
    cdef floating [:] delta_max_view = _delta_max

    
    for i in prange(total_target_tracers, nogil = True):
        for j in range(3):
            pos_view[i,j] = pos[0][i][j]
            vel_view[i,j] = vel[0][i][j]
        is_dm_view[i] = is_dm[0][i]
        is_attractor_view[i] = is_attractor[0][i]
        dweb_view[i] = dweb[0][i]
        delta_dm_view[i] = delta_dm[0][i]
        r_min_view[i] = r_min[0][i]
        delta_max_view[i] = delta_max[0][i]



    toret =  dict(pos = _pos, vel = _vel, is_dm = _is_dm, dweb = _dweb, delta_dm = _delta_dm, r_min = _r_min, delta_max = _delta_max, is_attractor = _is_attractor)
    print("Done")
    return toret#{key:np.array(val) for key, val in toret.items()}
    
            

cdef floating sinx(floating cos2x) noexcept nogil:
    if abs(abs(cos2x) - 1) < 1e-3:
        return 0
    else:
        return sqrt(1 -cos2x)



cdef floating coordinate_separation(floating a, floating b, floating box_size) noexcept nogil:
    cdef floating delta = abs(a - b)
    #cdef floating delta = (a - b)
    if (a-b) >= 0:
        return (delta if delta < 0.5*box_size else delta - box_size )
    else:
        return -(delta if delta < 0.5*box_size else delta - box_size )


cdef floating clip(floating x, floating min, floating max) noexcept nogil:
    if x < min:
        return min
    elif x > max:
        return max
    else:
        return x

cdef void collapse(floating[:] out_sat_pos, floating[:] sat_pos, floating[:] cen_pos, floating dist, 
                   double collapse_frac, double dist_upper_bound, 
                   floating[:] out_sat_vel, 
                   floating random_gauss_x, floating random_gauss_y, floating random_gauss_z, floating dm_at_cen, floating velocity_disp,
                   floating[:] box_size) noexcept nogil:

    
    cdef floating r = 0 #sqrt(dist)
    cdef floating cos_phi, cos_theta, sin_phi, sin_theta
    cdef size_t i
    
    for i in range(3):
        r += coordinate_separation[floating](sat_pos[i], cen_pos[i], box_size[i])**2
    r = sqrt(r)
    
    #if cen_pos.shape[0] > 3:
    #    dist -= (cen_pos[3] - sat_pos[3])**2
    #cdef floating r = sqrt(dist)


    if (r < <floating> dist_upper_bound) and (r > <floating> 1e-2):
        cos_phi = coordinate_separation[floating](sat_pos[2], cen_pos[2], box_size[2]) / r
        sin_phi = sinx[floating](cos_phi**2)
        #sin_phi = sin(acos(cos_phi))
        cos_theta = (coordinate_separation[floating](sat_pos[0], cen_pos[0], box_size[0]))/ (r * sin_phi) if sin_phi != 0 else sqrt(2) / 2
        #cos_theta = clip[floating](cos_theta, -1, 1)
        #if abs(cos_theta) > <floating> 1:
        #    printf("Found cos(theta) = %lf > 1 with x = %lf, r= %lf, sin(phi) = %lf\n ", cos_theta, coordinate_separation[floating](sat_pos[0], cen_pos[0], box_size[0]), r, sin_phi)
        #    abort()
        sin_theta = sinx[floating](cos_theta**2)
        #sin_theta = sin(acos(cos_theta))
        if (cos_phi != cos_phi) or (sin_phi != sin_phi) or (cos_theta != cos_theta) or (sin_theta != sin_theta):
            printf("%lf %lf, %lf, %lf, %lf\n", r, cos_phi, sin_phi, cos_theta, sin_theta)
            fflush(stdout)
            abort()

        r = r * <floating> collapse_frac

        out_sat_pos[0] = cen_pos[0] + r * cos_theta * sin_phi
        out_sat_pos[1] = cen_pos[1] + r * sin_theta * sin_phi
        out_sat_pos[2] = cen_pos[2] + r * cos_phi

        out_sat_vel[0] = out_sat_vel[0] + random_gauss_x * 1e1 * velocity_disp * (1 + (0 if dm_at_cen <= 0 else dm_at_cen))**0.5
        out_sat_vel[1] = out_sat_vel[1] + random_gauss_x * 1e1 * velocity_disp * (1 + (0 if dm_at_cen <= 0 else dm_at_cen))**0.5
        out_sat_vel[2] = out_sat_vel[2] + random_gauss_x * 1e1 * velocity_disp * (1 + (0 if dm_at_cen <= 0 else dm_at_cen))**0.5


    

cpdef dict subgrid_collapse(dict catalog, floating[:] params, floating[:] box_size, is_attractor_mask, size_t seed, size_t num_threads, debug = False):

    cdef mt19937 gen = mt19937(seed)
    cdef normal_distribution[floating] dist_gauss = normal_distribution[floating](0., 1.)
    #printf("Assigning DM particles to cells...")
    #fflush(stdout)
    #cdef vector[vector[Py_ssize_t]] mesh
    #mesh.reserve(grid_size**3)
    #mesh.resize(grid_size**3)
    #for i in range(grid_size**3):
    #    #dm_per_cell.push_back([])
    #    dm_per_cell.push_back(*(new vector[Py_ssize_t]()))

    cdef floating[:,:] pos_view, vel_view
    pos_view = catalog['pos']

    #for i in range(pos_view.shape[0]):
    #    ii = wrap_indices(<Py_ssize_t>(floor(grid_size * pos_view[i,0] / box_size[0])), grid_size)
    #    jj = wrap_indices(<Py_ssize_t>(floor(grid_size * pos_view[i,1] / box_size[1])), grid_size)
    #    kk = wrap_indices(<Py_ssize_t>(floor(grid_size * pos_view[i,2] / box_size[2])), grid_size)
    #    index_3d = INDEX(ii, jj, kk, grid_size, grid_size, grid_size)
    #    mesh[index_3d].push_back(i)
    
    #mask = catalog['is_dm'].astype(bool) & (catalog['dweb'] < 4)
    cdef cnp.ndarray[floating, ndim=2] attractors = catalog['pos'][is_attractor_mask]
    cdef cnp.ndarray[floating, ndim=2] not_attractors = catalog['pos'][~is_attractor_mask]
    cdef cnp.ndarray[floating, ndim=2] vel_attractors = catalog['vel'][is_attractor_mask]
    cdef cnp.ndarray[floating, ndim=2] vel_not_attractors = catalog['vel'][~is_attractor_mask]
    cdef cnp.ndarray[floating, ndim=1] dm = catalog['delta_dm'][is_attractor_mask]


    #cdef cnp.ndarray[floating, ndim=2] attractors_copy = attractors.copy()
    #cdef cnp.ndarray[floating, ndim=2] not_attractors_copy = not_attractors.copy()

    cdef floating[:,:] attractors_view, not_attractors_view, vel_attractors_view, vel_not_attractors_view#, attractors_copy_view, not_attractors_copy_view
    attractors_view = attractors
    not_attractors_view = not_attractors
    vel_attractors_view = vel_attractors
    vel_not_attractors_view = vel_not_attractors

    cdef cnp.ndarray[long, ndim=1] attractors_ids = catalog['order'][is_attractor_mask]
    cdef cnp.ndarray[long, ndim=1] not_attractors_ids = catalog['order'][~is_attractor_mask]
    
    



    cdef floating[:] dm_view = dm
    #attractors_copy_view = attractors_copy
    #not_attractors_copy_view = not_attractors_copy
    
    #tic = time.time()
    #tree = pybosque.Tree(not_attractors, idxs)
    tree = KDTree(attractors)#, boxsize = box_size)
    #print(f"Tree built in {time.time() - tic}s", flush=True)
    #tic = time.time()
    #r, ids = tree.query(attractors, 2, [0,1])
    #r, ids = tree.query(not_attractors, k = 2)
    dists, ids = tree.query(attractors, k = 2, eps = 0., distance_upper_bound = None, sqr_dists = True)#, workers = -1)
    cdef floating[:] dists_view
    dists_view = dists[:,1]
    cdef unsigned[:] ids_view
    ids_view = ids[:,1]
    #ids is the array of ids in not_attractors that are closest to eah attractor
    #print(f"Tree query in {time.time() - tic}s", flush=True)
    cdef Py_ssize_t i
    cdef floating collapse_frac, gauss_rand
    
    tic = time.time()
    for i in prange(ids_view.shape[0], nogil = True, num_threads = num_threads):#, use_threads_if=ids_view.shape[0] > 1e6):
        #gauss_rand = dist_gauss(gen)
        collapse_frac = params[0] #* (1 + gauss_rand * 0.5) if gauss_rand > 0 else exp(gauss_rand * 0.5)
        collapse[floating](attractors_view[ids_view[i],:], attractors_view[ids_view[i],:], attractors_view[i,:], dists_view[i], params[0], params[1], 
                                                            vel_attractors_view[ids_view[i],:], dist_gauss(gen), dist_gauss(gen), dist_gauss(gen), dm_view[i], params[4],
                                                            box_size)
    #print(f"Collapse in {time.time() - tic}s", flush=True)
    if debug:
        print(dists)
        tic = time.time()
        tree = KDTree(attractors)#, boxsize = box_size)
        print(f"Tree built in {time.time() - tic}s", flush=True)
        tic = time.time()
        dists, ids = tree.query(attractors, k = 2, eps = 0., distance_upper_bound = None, sqr_dists = True)#, workers = -1)
        print(dists)


    
    #log_delta_dm = (catalog['delta_dm'])

    #rescaled_delta_dm = box_size[0] * (log_delta_dm  - log_delta_dm.min()) / (log_delta_dm.max()  - log_delta_dm.min())
    #attractors = np.c_[attractors,rescaled_delta_dm[is_attractor_mask]]
    #not_attractors = np.c_[not_attractors,rescaled_delta_dm[~is_attractor_mask]]
    #attractors_copy = attractors.copy()
    #not_attractors_copy = not_attractors.copy()


    attractors_view = attractors
    not_attractors_view = not_attractors
    #attractors_copy_view = attractors_copy
    #not_attractors_copy_view = not_attractors_copy

    #tic = time.time()
    tree = KDTree(attractors)#, boxsize = box_size)
    #print(f"Tree built in {time.time() - tic}s", flush=True)
    #tic = time.time()
    dists, ids = tree.query(not_attractors, k = 1, eps = 0., distance_upper_bound = None, sqr_dists = True)#, workers = -1)
    dists_view = dists
    ids_view = ids
    mask = np.isfinite(dists)
    #dist_corr = (not_attractors[mask,3] - attractors[ids[mask],3])**2
    
    #tic = time.time()
    
    
    for i in prange(ids_view.shape[0], nogil = True):#, use_threads_if=ids_view.shape[0] > 1e6):
        #gauss_rand = dist_gauss(gen)
        collapse_frac = params[2]
        collapse[floating](not_attractors_view[i,:], not_attractors_view[i,:], attractors_view[ids_view[i],:], dists_view[i], collapse_frac , params[3], 
                                                      vel_not_attractors_view[i,:], dist_gauss(gen), dist_gauss(gen), dist_gauss(gen), dm_view[ids_view[i]], params[4],
                                                      box_size)
    #print(f"Collapse in {time.time() - tic}s", flush=True)
    if debug:
        #print(dists[mask] - dist_corr)
        #tic = time.time()
        tree = KDTree(attractors)#, boxsize = box_size)
        #print(f"Tree built in {time.time() - tic}s", flush=True)
        #tic = time.time()
        dists, ids = tree.query(not_attractors, k = 1, eps = 0., distance_upper_bound = None, sqr_dists = True)#, workers = -1)
        mask = np.isfinite(dists)
        #dist_corr = (not_attractors[mask,3] - attractors[ids[mask],3])**2
        #print(dists[mask] - dist_corr)

    catalog['pos'] = np.vstack((attractors[:,:3], not_attractors[:,:3]))
    catalog['vel'] = np.vstack((vel_attractors[:,:3], vel_not_attractors[:,:3]))
    catalog['order'] = np.concatenate((attractors_ids, not_attractors_ids))

    return catalog


cpdef dict single_collapse_step(cnp.ndarray[floating, ndim=2] attractors, cnp.ndarray[floating, ndim=2] vel_attractors, cnp.ndarray[floating, ndim=1] dm,
                                cnp.ndarray[floating, ndim=2] not_attractors, cnp.ndarray[floating, ndim=2] vel_not_attractors,
                                floating[:] params, 
                                size_t k_neighbour,
                                floating[:] box_size, size_t seed, size_t num_threads, debug = False):
    #params = (collapse_frac, collapse_radius, velocity_dispersion)

    cdef mt19937 gen = mt19937(seed)
    cdef normal_distribution[floating] dist_gauss = normal_distribution[floating](0., 1.)
    

    #cdef cnp.ndarray[floating, ndim=2] attractors_copy = attractors.copy()
    #cdef cnp.ndarray[floating, ndim=2] not_attractors_copy = not_attractors.copy()

    cdef floating[:,:] attractors_view, not_attractors_view, vel_attractors_view, vel_not_attractors_view#, attractors_copy_view, not_attractors_copy_view
    attractors_view = attractors
    not_attractors_view = not_attractors
    vel_attractors_view = vel_attractors
    vel_not_attractors_view = vel_not_attractors


    cdef floating[:] dm_view = dm
    #attractors_copy_view = attractors_copy
    #not_attractors_copy_view = not_attractors_copy
    
    #tic = time.time()
    #tree = pybosque.Tree(not_attractors, idxs)
    tree = KDTree(attractors)#, boxsize = box_size)
    #print(f"Tree built in {time.time() - tic}s", flush=True)
    #tic = time.time()
    #r, ids = tree.query(attractors, 2, [0,1])
    #r, ids = tree.query(not_attractors, k = 2)
    dists, ids = tree.query(not_attractors, k = k_neighbour, eps = 0., distance_upper_bound = None, sqr_dists = True)#, workers = -1)
    cdef floating[:] dists_view
    cdef unsigned[:] ids_view
    if k_neighbour > 1:
        dists_view = dists[:,k_neighbour-1]
        ids_view = ids[:,k_neighbour-1]
    else:
        dists_view = dists
        ids_view = ids
    #ids is the array of ids in not_attractors that are closest to eah attractor
    #print(f"Tree query in {time.time() - tic}s", flush=True)
    cdef Py_ssize_t i
    cdef floating collapse_frac, gauss_rand

    for i in prange(ids_view.shape[0], nogil = True, num_threads = num_threads):#, use_threads_if=ids_view.shape[0] > 1e6):
        collapse[floating](not_attractors_view[i,:], not_attractors_view[i,:], attractors_view[ids_view[i],:], dists_view[i], params[0] , params[1], 
                                                      vel_not_attractors_view[i,:], dist_gauss(gen), dist_gauss(gen), dist_gauss(gen), dm_view[ids_view[i]], params[2],
                                                      box_size)
    
    #log_delta_dm = (catalog['delta_dm'])

    #rescaled_delta_dm = box_size[0] * (log_delta_dm  - log_delta_dm.min()) / (log_delta_dm.max()  - log_delta_dm.min())
    #attractors = np.c_[attractors,rescaled_delta_dm[is_attractor_mask]]
    #not_attractors = np.c_[not_attractors,rescaled_delta_dm[~is_attractor_mask]]
    #attractors_copy = attractors.copy()
    #not_attractors_copy = not_attractors.copy()
    catalog = {}
    catalog['pos'] = np.vstack((attractors[:,:3], not_attractors[:,:3]))
    catalog['vel'] = np.vstack((vel_attractors[:,:3], vel_not_attractors[:,:3]))

    return catalog


cdef void move_centers(floating step_size, int i, int j, int k, floating[:] pos, floating[:] field, int grid_size, floating[:] box_size) noexcept nogil:


    cdef floating d_dx = 0
    cdef floating d_dy = 0
    cdef floating d_dz = 0
    cdef Py_ssize_t ll


    for ll in range(-1, 1, 2):
        index_3d = INDEX(wrap_indices(i+ll, grid_size),
                                wrap_indices(j, grid_size),
                                wrap_indices(k, grid_size),
                                grid_size,
                                grid_size,
                                grid_size);
        d_dx += <floating> ll * field[index_3d]
    
    pos[0] += step_size * d_dx * grid_size / (2 * box_size[0])


    for ll in range(-1, 1, 2):
        index_3d = INDEX(wrap_indices(i, grid_size),
                                wrap_indices(j+ll, grid_size),
                                wrap_indices(k, grid_size),
                                grid_size,
                                grid_size,
                                grid_size);
        d_dy += <floating> ll * field[index_3d]
    
    pos[1] += step_size * d_dy * grid_size / (2 * box_size[1])



    for ll in range(-1, 1, 2):
        index_3d = INDEX(wrap_indices(i, grid_size),
                                wrap_indices(j, grid_size),
                                wrap_indices(k+ll, grid_size),
                                grid_size,
                                grid_size,
                                grid_size);
        d_dz += <floating> ll * field[index_3d]
    
    pos[2] += step_size * d_dz * grid_size / (2 * box_size[2])



        
