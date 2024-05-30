#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <cstdlib>


#define INDEX(i,j,k, nx, ny, nz)  (k + j*nz + i*ny*nz)


// Define the SubgridCatalog structure
struct SubgridCatalog {
    std::vector<std::vector<double>> pos;
    std::vector<std::vector<double>> vel;
    std::vector<bool> is_dm;
    std::vector<double> dweb;
    std::vector<double> delta_dm;
    std::vector<double> r_min;
    std::vector<double> delta_max;
    std::vector<bool> attractor;
    float param1;
    float param2;
    size_t num_particles;
};

int wrap_indices(int value, int size);


template <typename T> T weights(T ddx, T ddy, T ddz, T ii, T jj, T kk)
{
    return (((1 - ddx) + ii * (-1 + 2 * ddx)) * 
            ((1 - ddy) + jj * (-1 + 2 * ddy)) *
            ((1 - ddz) + kk * (-1 + 2 * ddz)));
}



// Define read_cic function

template<typename T> T read_cic(const std::vector<T>& field, 
                const std::vector<T>& position, 
                const std::vector<T>& box_size, 
                const std::vector<T>& box_min,
                const std::vector<int>& dims,
                bool wrap) ;

template <typename T> T read_cic(const std::vector<T>& field, 
                const std::vector<T>& position, 
                const std::vector<T>& box_size, 
                const std::vector<T>& box_min,
                const std::vector<int>& dims,
                bool wrap) 
{
    std::vector<T> cell_size = {box_size[0] / dims[0], box_size[1] / dims[1], box_size[2] / dims[2]};

    T xpos = (position[0] - box_min[0]) / cell_size[0];
    T ypos = (position[1] - box_min[1]) / cell_size[1];
    T zpos = (position[2] - box_min[2]) / cell_size[2];

    int i = static_cast<int>(std::floor(xpos));
    int j = static_cast<int>(std::floor(ypos));
    int k = static_cast<int>(std::floor(zpos));

    T ddx = xpos - i;
    T ddy = ypos - j;
    T ddz = zpos - k;
    T result = 0;
    for (size_t ii = 0; ii < 2; ii++){
        for (size_t jj = 0; jj < 2; jj++){
            for (size_t kk = 0; kk < 2; kk++){
                size_t index_3d = INDEX(wrap_indices(i+ii, dims[0]), 
                                        wrap_indices(j+jj, dims[1]), 
                                        wrap_indices(k+kk, dims[2]),
                                        dims[0],
                                        dims[1],
                                        dims[2]);
                result += weights<T>(ddx, ddy, ddz, ii, jj, kk) * field[index_3d];
            }
        }
    }

    return result;
}


bool is_attractor_fun(bool is_dm, double cw_type);
SubgridCatalog assign_particles_to_gals(
    const std::vector<std::vector<double>>& dm_particles, 
    const std::vector<unsigned int>& target_ncount, 
    int grid_x, int grid_y, int grid_z,
    const std::vector<double>& box_size,
    const std::vector<double>& box_min,
    const std::vector<double>& dm_cw_type,
    const std::vector<double>& dm_dens,
    const std::vector<std::vector<double>>& displacement,
    const std::vector<std::vector<double>>& velocities,
    double dist,
    bool debug);
    