#ifndef KERNELS_H
#define KERNELS_H
#include <vector>
#include <cmath>
double gaussian_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths);

double linf_triangle_kernel(const double* s1, const double* s2, 
                    const std::vector<unsigned int>& dim, 
                    const double* widths);

#endif
