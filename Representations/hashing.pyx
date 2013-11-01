cimport numpy as np
import numpy as np
cimport cython

cdef int BIG_INT = 2147483647

@cython.boundscheck(False)
@cython.cdivision(True)
cdef long hash(long [:] A, long [:] R, int increment, int max):
    cdef int i = 0
    cdef int n = R.shape[0]
    cdef int l = A.shape[0]
    cdef int ind
    cdef long result = 0
    for i in range(l):
        ind = (A[i] + (increment * i)) % n
        result += R[ind]
    result = result % max
    return result

@cython.boundscheck(False)
@cython.cdivision(True)
def physical_addr(np.ndarray[np.npy_long, ndim=1] A,
                   np.ndarray[np.npy_long, ndim=1] R,
                   np.ndarray[np.npy_longlong, ndim=1] counts,
                   np.ndarray[np.npy_longlong, ndim=1] check_data):
    """
    Map a virtual vector address A to a physical address i.e. the actual
    feature number.
    This is the actual hashing trick
    """
    cdef long [:] A_view = A
    cdef long [:] R_view = R

    cdef int h1 = hash(A_view, R_view, 449, R.shape[0])
    cdef int check_val = hash(A_view, R_view, 457, BIG_INT)

    if counts[h1] == 0:
        check_data[h1] = check_val
        counts[h1] = counts[h1] + 1
        return h1, 0
    elif np.all(check_val == check_data[h1]):
        # clear hit, everything's fine
        counts[h1] = counts[h1] + 1
        return h1, 0
    else:
        return h1, 1

