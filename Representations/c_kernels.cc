#include "c_kernels.h"
#include <cmath> 

double gaussian_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths) {
    double exponent = 0;
    for (unsigned int d : dim) {
        exponent += - pow((s1[d] - s2[d]) / widths[d], 2);
    }
    return exp(exponent);

}

double linf_triangle_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths) {
    double res = 1, r;
    for (unsigned int d : dim) {

        r = 1 - std::abs(s1[d] - s2[d]) / widths[d];
        if (r <= 0) 
            return 0;
        else 
            res = fmin(r, res);
    }
    return res;

}
