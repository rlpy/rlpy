#include "c_kernels.h"
#include <cmath> 

double gaussian_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths) {
    double exponent = 0;
    unsigned int d = 0;
    
    //for (d : dim) {
    // instead of the elegant solution above we need to do the cryptic fuck-up
    // below to avoid dependency on C++11 :-(
    for(std::vector<unsigned int>::const_iterator it = dim.begin(); 
            it != dim.end(); ++it) {
        d = *it;
        exponent += - pow((s1[d] - s2[d]) / widths[d], 2);
    }
    return exp(exponent);

}

double linf_triangle_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths) {
    double res = 1, r;
    unsigned int d = 0;
    for(std::vector<unsigned int>::const_iterator it = dim.begin(); 
            it != dim.end(); ++it) {
        d = *it;

    //for (unsigned int d : dim) {

        r = 1 - std::abs(s1[d] - s2[d]) / widths[d];
        if (r <= 0) 
            return 0;
        else 
            res = std::min(r, res);
    }
    return res;

}
