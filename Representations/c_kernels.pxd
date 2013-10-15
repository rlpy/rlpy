from libcpp.vector cimport vector
cdef extern from "c_kernels.h":
    cdef double gaussian_kernel(double* s1, double* s2, vector[unsigned int] dim, double* widths)
    cdef double linf_triangle_kernel(double* s1, double* s2, vector[unsigned int] dim, double* widths)
