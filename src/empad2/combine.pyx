# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

cimport cython
from cython.parallel import prange, parallel
from cython.cimports.libc.stdlib import malloc, free
from libc.math cimport floor
from cython.cimports.libc.limits import INT_MIN
import numpy as np
cimport numpy as cnp

cdef int clip(int x, int min, int max) nogil:
    if (x<min): 
        x = min
    if (x>max): 
        x = max
    return x

cdef int argmax(cnp.npy_uint64 *x, int N) nogil:
    """
    find index of maximum value of integer array
    int *x: pointer to array
    int N: length of array
    """
    cdef int a = 0
    cdef cnp.npy_uint64 maxval = 0

    for i in range(N):
        if x[i] > maxval:
            a = i
            maxval = x[i]

    return a

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def combine_quadratic(
    float[:,:,:,::] datacube,
    float[:,:,::] Ml,
    float[:,:,::] alpha,
    float[:,:,::] Md,
    float[:,:,::] Oh,
    float[:,:,::] Ot,
):
    '''
    Combine using quadratic method
    '''

    # shape of the 4D array as a C array
    cdef Py_ssize_t shape[4]
    shape[:] = [datacube.shape[0], datacube.shape[1], datacube.shape[2], datacube.shape[3]]

    # iteration variables
    cdef Py_ssize_t i,j,k,l, ij

    # working variables
    cdef cnp.npy_uint32 data, analog_int, digital_int
    cdef float analog, digital, gain_bit, analog_x_gain_bit
    # cdef bool gain_bit

    # loop is parallelized across all patterns, with the first two
    # indices rolled for better division of labor
    for ij in prange(shape[0] * shape[1], nogil=True):
        i = ij / shape[1]
        j = ij % shape[1]

        # combine each pixel
        for k in range(shape[2]):
            for l in range(shape[3]):

                data = (<cnp.npy_uint32 *>(&datacube[i,j,k,l]))[0] # view the data as a uint32 (?)
                analog_int = data & <cnp.npy_uint32>0x3FFF
                digital_int = (data & <cnp.npy_uint32>0x3FFFC000) >> 14

                analog = <float>analog_int
                digital = <float>digital_int
                gain_bit = <float>((data & <cnp.npy_uint32>0x80000000) >> 31)

                analog_x_gain_bit = analog * gain_bit # premultiply

                datacube[i,j,k,l] = (
                    analog * (1.0 - gain_bit) # analog part
                    + Ml[j % 2, k,l] * analog_x_gain_bit  # ml
                    + alpha[j % 2, k,l] * analog_x_gain_bit * analog_x_gain_bit  # alpha
                    + Md[j % 2, k,l] * digital  # md
                    + Oh[j % 2, k,l] * gain_bit  # oh
                    - Ot[j % 2, k,l]  # ot
                )

                # things for debugging:
                # datacube[i,j,k,l] = analog
                # datacube[i,j,k,l] = digital
                # datacube[i,j,k,l] = gain_bit



@cython.boundscheck(False)
@cython.wraparound(False)
# @cython.cdivision(True)
def combine_quadratic_bgsub_debounce(
    float[:,:,:,::] datacube,
    float[:,:,::] Ml,
    float[:,:,::] alpha,
    float[:,:,::] Md,
    float[:,:,::] Oh,
    float[:,:,::] Ot,
    float[:,:,::] FF,
    float[:,:,::] background,
    float debounce_min = -200.0,
    float debounce_max = 220.0,
    int debounce_bins = 420,
):
    '''
    Combine using quadratic method, with background subtraction and debouncing
    '''

    debounce_values_npy = np.zeros((datacube.shape[0], datacube.shape[1]), dtype=np.float32)
    cdef cnp.npy_float32[:,::] debounce_values = debounce_values_npy

    # shape of the 4D array as a C array
    cdef Py_ssize_t shape[4]
    shape[:] = [datacube.shape[0], datacube.shape[1], datacube.shape[2], datacube.shape[3]]

    # iteration variables
    cdef Py_ssize_t i,j,k,l, ij

    # working variables
    cdef cnp.npy_uint32 data, analog_int, digital_int
    cdef float analog, digital, gain_bit, analog_x_gain_bit, combined_data
    cdef float debounce_correction

    # histogram accumulator
    cdef cnp.npy_uint64 *histogram
    cdef float accumulator_factor, histogram_factor
    accumulator_factor = <float>debounce_bins / ( <float>debounce_max - <float>debounce_min)
    histogram_factor = (<float>debounce_max - <float>debounce_min) / <float>debounce_bins
    cdef int hist_idx, h

    # loop is parallelized across all patterns, with the first two
    # indices rolled for better division of labor
    with nogil, parallel():
        # allocate scratch for histogram accumulation in each thread
        histogram = <cnp.npy_uint64 *>malloc((debounce_bins+2)*sizeof(cnp.npy_uint64))

        for ij in prange(shape[0] * shape[1]):
            i = ij // shape[1]
            j = ij % shape[1]

            # clear the histogram accumulator
            for h in range(debounce_bins+2):
                histogram[h] = 0

            # combine each pixel
            for k in range(shape[2]):
                for l in range(shape[3]):

                    data = (<cnp.npy_uint32 *>(&datacube[i,j,k,l]))[0] # view the data as a uint32 (?)
                    analog_int = data & <cnp.npy_uint32>0x3FFF
                    digital_int = (data & <cnp.npy_uint32>0x3FFFC000) >> 14

                    analog = <float>analog_int
                    digital = <float>digital_int
                    gain_bit = <float>((data & <cnp.npy_uint32>0x80000000) >> 31)

                    analog_x_gain_bit = analog * gain_bit # premultiply

                    combined_data = (
                        analog * (1.0 - gain_bit) # analog part
                        + Ml[j % 2, k,l] * analog_x_gain_bit  # ml
                        + alpha[j % 2, k,l] * analog_x_gain_bit * analog_x_gain_bit  # alpha
                        + Md[j % 2, k,l] * digital  # md
                        + Oh[j % 2, k,l] * gain_bit  # oh
                        - Ot[j % 2, k,l]  # ot
                        - background[j % 2, k,l]
                    )

                    datacube[i,j,k,l] = combined_data

                    # accumulate histogram data
                    # (the first and last index are for values out of the histogram range)
                    hist_idx = 1 + <int>floor((combined_data - <float>debounce_min) * accumulator_factor)
                    hist_idx = clip(hist_idx, 0, debounce_bins+1)
                    histogram[hist_idx] += 1

            # compute debounce offset from histogram (after clearing invalid bins)
            # currently this is just finding the maximum, ideally this should do 
            # some sort of fitting
            histogram[0] = 0
            histogram[debounce_bins + 1] = 0
            hist_idx = argmax(histogram, debounce_bins + 2)
            debounce_correction = (<float>hist_idx - 0.5) * histogram_factor + <float>debounce_min

            debounce_values[i,j] = <cnp.npy_float32>debounce_correction

            # apply debounce, background, and flatfield
            for k in range(shape[2]):
                for l in range(shape[3]):
                    combined_data = datacube[i,j,k,l]

                    datacube[i,j,k,l] = (combined_data - debounce_correction) * FF[j%2,k,l]

        # deallocate scratch array
        free(histogram)

    return debounce_values_npy
