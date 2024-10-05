# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp

# simple build command for high optimization level:
# CFLAGS=-I$(python -c "import numpy;print(numpy.get_include() + ' -O3 -march=native -fopenmp')") cythonize -f -i combine.pyx
#

cimport cython
from cython.parallel import prange, parallel
from cython.cimports.libc.stdlib import malloc, free
from libc.math cimport floor
from cython.cimports.libc.limits import INT_MIN
import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_lapack cimport sgels


# extra size for work array used in sgels
# (3 is probably alright, but hey memory is cheap)
cdef int WORK_ARRAY_FACTOR = 6

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

cdef int fill_fit_coords(
    cnp.npy_float32 *fit_coords, 
    cnp.npy_float32 *hist_values,
    cnp.npy_uint64 *histogram,
    const int hist_max,
    const float debounce_min,
    const float debounce_max,
    const int debounce_bins,
    const int fit_window,
) nogil:
    """
    Fill *fit_coords with the values for least squares polynomial fitting of the
    histogram values around index `hist_max`.

    `fit_coords` should be size 3 * ((fit_window * 2) + 1)
    """

    cdef int i, j, s
    cdef cnp.npy_float32 bin_center
    s = 2*fit_window + 1 # stride for fit_coords
    for i in range(s):
        # fill in the coordinates array with [1, bin_center, bin_center**2]
        bin_center = (
            (<float>(hist_max + i - fit_window) + 0.5) 
            * (debounce_max-debounce_min)/<float>debounce_bins
            + debounce_min
        )

        fit_coords[i] = 1.0
        fit_coords[s + i] = bin_center
        fit_coords[2*s + i] = bin_center * bin_center

        # fill in the cropped histogram with the values
        j = hist_max + i - fit_window
        hist_values[i] = <cnp.npy_float32>histogram[j] if (j >= 0 and j < debounce_bins) else 0.0  

    return 0

cdef float fit_histogram_peak(
    cnp.npy_float32 *fit_coords,
    cnp.npy_float32 *hist_values,
    cnp.npy_float32 *work_array,
    int work_array_size,
    const int fit_window,
) nogil:

    cdef int status, N_coords, N, NRHS
    cdef char trans = 'N'
    N_coords = (fit_window * 2) + 1
    N = 3
    NRHS = 1

    # get the edges of the fit range
    cdef float window_min = fit_coords[N_coords]
    cdef float window_max = fit_coords[2*N_coords-1]

    # cdef float test = fit_coords[N_coords]
    cdef float test = hist_values[fit_window]
    
    # least squares, see https://netlib.org/lapack/explore-html/d8/d83/group__gels_gadc3f6f560a228cfb5ad6c7456da1b778.html#gadc3f6f560a228cfb5ad6c7456da1b778
    sgels(
        &trans,                 # TRANS [in]
        &N_coords,              # M [in]
        &N,                     # N [in]
        &NRHS,                  # NRHS [in]
        fit_coords,             # A [in,out]
        &N_coords,              # LDA [in]
        hist_values,            # B [in,out]
        &N_coords,              # LDB [in]
        work_array,             # WORK [out]
        &work_array_size,       # LWORK [in]
        &status,                # INFO [out]
    )

    cdef float polymax 
    if hist_values[2] != 0.0:
        polymax = -hist_values[1] / 2.0 / hist_values[2]
    else:
        polymax = -1000.0
    return 0.0 if (polymax > window_max or polymax < window_min) else polymax

    # return test
    # return <float>status
    # return work_array[0]
    # return hist_values[2]


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
    const float[:,:,::] Ml,
    const float[:,:,::] alpha,
    const float[:,:,::] Md,
    const float[:,:,::] Oh,
    const float[:,:,::] Ot,
    const float[:,:,::] FF,
    const float[:,:,::] background,
    const float debounce_min = -200.0,
    const float debounce_max = 220.0,
    const int debounce_bins = 420,
    const int fit_window = 5,
    const bint polyfit_histogram_peak = False,
):
    '''
    Combine using quadratic method, with background subtraction and debouncing.
    `datacube` is modified in place
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
    cdef cnp.npy_float32 *fit_coords 
    cdef cnp.npy_float32 *hist_values
    cdef cnp.npy_float32 *work_array
    cdef int work_array_size = (2*fit_window+1)*3*WORK_ARRAY_FACTOR
    cdef float accumulator_factor, histogram_factor
    accumulator_factor = <float>debounce_bins / ( <float>debounce_max - <float>debounce_min)
    histogram_factor = (<float>debounce_max - <float>debounce_min) / <float>debounce_bins
    cdef int hist_idx, h

    # loop is parallelized across all patterns, with the first two
    # indices rolled for better division of labor
    with nogil, parallel():
        # allocate scratch for histogram accumulation in each thread
        histogram = <cnp.npy_uint64 *>malloc((debounce_bins+2)*sizeof(cnp.npy_uint64))

        # allocate stuff for the polynomial fit
        # 3 * 2N+1 coordinates
        fit_coords = <cnp.npy_float32 *>malloc(3*(2*fit_window+1)*sizeof(cnp.npy_float32))
        hist_values = <cnp.npy_float32 *>malloc((2*fit_window+1)*sizeof(cnp.npy_float32))
        work_array = <cnp.npy_float32 *>malloc(work_array_size*sizeof(cnp.npy_float32))

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

            if polyfit_histogram_peak:
                # fill in array with coordinates for fitting
                fill_fit_coords(
                    fit_coords,
                    hist_values,
                    histogram,
                    hist_idx,
                    debounce_min,
                    debounce_max,
                    debounce_bins,
                    fit_window,
                )
                # perform least squares polynomial fit to peak
                debounce_correction = fit_histogram_peak(
                    fit_coords,
                    hist_values,
                    work_array,
                    work_array_size,
                    fit_window,
                )
            else:
                debounce_correction = (<float>hist_idx - 0.5) * histogram_factor + <float>debounce_min

            debounce_values[i,j] = <cnp.npy_float32>debounce_correction

            # apply debounce, background, and flatfield
            for k in range(shape[2]):
                for l in range(shape[3]):
                    combined_data = datacube[i,j,k,l]

                    datacube[i,j,k,l] = (combined_data - debounce_correction) * FF[j%2,k,l]

        # deallocate scratch arrays
        free(histogram)
        free(fit_coords)
        free(hist_values)
        free(work_array)

    return debounce_values_npy
