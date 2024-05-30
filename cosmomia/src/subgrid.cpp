#include "subgrid.h"




// Define wrap_indices function
int wrap_indices(int value, int size) {
    if (value < 0) return size + value;
    if (value >= size) return value - size;
    return value;
}




// Define is_attractor_fun function
//std::vector<bool> is_attractor_fun(const std::vector<bool>& is_dm, const std::vector<double>& cw_type) {
//    std::vector<bool> attractor(is_dm.size());
//    for (size_t i = 0; i < is_dm.size(); ++i) {
//        attractor[i] = (cw_type[i] < 4) && is_dm[i];
//    }
//    return attractor;
//}
bool is_attractor_fun(bool is_dm, double cw_type){
    return is_dm && (cw_type < 4);
}

// Define the assign_particles_to_gals function
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
    bool debug)
{
    std::vector<std::vector<unsigned int>> dm_per_cell(grid_x * grid_y * grid_z);
    std::vector<std::vector<double>> pos;
    std::vector<std::vector<double>> vel;
    std::vector<bool> is_dm;
    std::vector<bool> is_attractor;
    std::vector<double> dweb;
    std::vector<double> delta_dm;
    std::vector<double> r_min;
    std::vector<double> delta_max;
    size_t used_dm = 0;
    size_t sampled_new = 0;

    std::vector<int> dims = {grid_x, grid_y, grid_z};
    std::vector<double> bin_size = {box_size[0] / grid_x, box_size[1] / grid_y, box_size[2] / grid_z};

    for (size_t i = 0; i < dm_particles[0].size(); ++i) {
        int idx = wrap_indices(static_cast<int>(std::floor(grid_x * dm_particles[0][i] / box_size[0])), grid_x);
        int idy = wrap_indices(static_cast<int>(std::floor(grid_y * dm_particles[1][i] / box_size[1])), grid_y);
        int idz = wrap_indices(static_cast<int>(std::floor(grid_z * dm_particles[2][i] / box_size[2])), grid_z);
        int index_3d = idx + grid_x * (idy + grid_y * idz);
        dm_per_cell[index_3d].push_back(i);
    }
    std::vector<double> grid_center(3);
    for (int i = 0; i < grid_x; ++i) {
        for (int j = 0; j < grid_y; ++j) {
            for (int k = 0; k < grid_z; ++k) {
                unsigned int number_cen_in_cell = target_ncount[i + grid_x * (j + grid_y * k)];
                if (number_cen_in_cell == 0) continue;

                double grid_center_x = (i + 0.5) * (box_size[0] / grid_x);
                double grid_center_y = (j + 0.5) * (box_size[1] / grid_y);
                double grid_center_z = (k + 0.5) * (box_size[2] / grid_z);

                grid_center[0] = grid_center_x;
                grid_center[1] = grid_center_y;
                grid_center[2] = grid_center_z;

                int index_3d = i + grid_x * (j + grid_y * k);
                std::vector<unsigned int>& dm_particles_in_cell = dm_per_cell[index_3d];
                size_t number_dm_in_cell = dm_particles_in_cell.size();

                for (unsigned int par = 0; par < number_cen_in_cell; ++par) {
                    std::vector<double> new_pos(3);

                    if (par < number_dm_in_cell) {
                        pos.push_back({dm_particles[0][dm_particles_in_cell[par]],
                                       dm_particles[1][dm_particles_in_cell[par]],
                                       dm_particles[2][dm_particles_in_cell[par]]});
                        is_dm.push_back(true);
                        used_dm++;
                    } 
                    else {
                        for (int ax = 0; ax < 3; ++ax) {
                            if (number_dm_in_cell > 0) {
                                unsigned int idx = dm_particles_in_cell[par % number_dm_in_cell];
                                double gc = dm_particles[ax][idx];
                                gc = fmod(gc + box_size[ax], box_size[ax]);
                                new_pos[ax] = fmod(gc + ((static_cast<double>(rand()) / RAND_MAX) - 0.5) * bin_size[ax], box_size[ax]);
                            } 
                            else {
                                new_pos[ax] = fmod(grid_center[ax] + ((static_cast<double>(rand()) / RAND_MAX) - 0.5) * bin_size[ax], box_size[ax]);
                            }
                        }
                    }
                    dweb.push_back(dm_cw_type[index_3d]);
                    delta_dm.push_back(read_cic(dm_dens, pos[pos.size()-1], box_size, box_min, dims, true));
                    std::vector<double> _vel = {
                        read_cic(velocities[0], pos[pos.size()-1], box_size, box_min, dims, true),
                        read_cic(velocities[1], pos[pos.size()-1], box_size, box_min, dims, true),
                        read_cic(velocities[2], pos[pos.size()-1], box_size, box_min, dims, true)
                    };
                    vel.push_back(_vel);
                    is_attractor.push_back(is_attractor_fun(is_dm[is_dm.size()-1], dweb[dweb.size()-1]));
                    r_min.push_back(1e10);
                    delta_max.push_back(-1e10);
                }
            }
        }
    }
    SubgridCatalog catalog = {pos, vel, is_dm, dweb, delta_dm, r_min, delta_max, is_attractor, 0, 1};
    return catalog;
}


