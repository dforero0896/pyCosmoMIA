#include "utils.h"

std::discrete_distribution<int> setup_discrete_distribution(std::vector<double> bin_counts){
    std::discrete_distribution<int> dist_disc(bin_counts.begin(), bin_counts.end());
    return dist_disc;
}

