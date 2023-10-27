import py4DSTEM
import numpy as np
import os
from pathlib import Path

__all__ = ["load_background", "load_dataset"]


def load_background():
    pass

def load_dataset():
    pass

##############################################################

# Since only one sensor exists at the moment I won't worry about how to handle different calibration files!
calibration_path = Path(__file__).parent / "calibration_data" / "SENSOR_001"
cal_names = ["G1A", "G2A", "G1B", "G2B", "FFA", "FFB", "B2A", "B2B"]
_G1A, _G2A, _G1B, _G2B, _FFA, _FFB, _B2A, _B2B = [
    np.fromfile(
        os.path.join(
            calibration_path,
            f"{calname}.r32",
        ),
        dtype=np.float32,
        count=128 * 128,
    ).reshape(128, 128)
    for calname in cal_names
]

# Constant data used in debouncing
binwidth = 10
histogram_bins, binwidth = np.linspace(-200-binwidth//2, 220-binwidth//2, num=420//binwidth + 1, retstep=True)
# get coordinates for doing polynomial fit
bin_centers = (histogram_bins[:-1] + histogram_bins[1:]) / 2.
bin_centers_sq = bin_centers ** 2
fit_coords = np.stack((np.ones_like(bin_centers),bin_centers,bin_centers_sq),axis=1)

def _debounce_frame(frame, fit_range=5):
    hist,edges = np.histogram(frame.flat, bins=histogram_bins)
    hist_peak = np.argmax(hist)
    
    if hist_peak < fit_range or hist_peak > len(hist)-fit_range-1:
        return 0.0

    polyfit = np.linalg.lstsq(
        fit_coords[hist_peak-fit_range:hist_peak+fit_range], 
        hist[hist_peak-fit_range:hist_peak+fit_range], 
        rcond=None,
    )[0]
    
    polymax = -polyfit[1]/2/polyfit[2]

    # make sure the peak is inside the fit range, else return 0.0
    return polymax if (polymax > bin_centers[hist_peak - fit_range] and polymax < bin_centers[hist_peak + fit_range]) else 0.0

def _process_EMPAD2_datacube(
    datacube, 
    background_even=None, 
    background_odd=None,
    ) -> None:
    # You might think that "even" indices are 0, 2, 4, etc...
    # but you would be wrong, because the original code was 
    # written in MATLAB
    background = background_even is not None and background_odd is not None

    # apply calibration to each pattern
    for rx, ry in py4DSTEM.tqdmnd(datacube.data.shape[0], datacube.data.shape[1]):
        data = datacube.data[rx, ry].view(np.uint32)
        analog = np.bitwise_and(data, 0x3FFF).astype(np.float32)
        digital = np.right_shift(np.bitwise_and(data, 0x3FFFC000), 14).astype(
            np.float32
        )
        gain_bit = np.right_shift(np.bitwise_and(data, 0x80000000), 31)

        if ry % 2:
            # "even" frame
            datacube.data[rx, ry] = (
                analog * (1 - gain_bit)
                + _G1B * (analog - _B2B) * gain_bit
                + _G2B * digital
            )
        else:
            # "odd" frame
            datacube.data[rx, ry] = (
                analog * (1 - gain_bit)
                + _G1A * (analog - _B2A) * gain_bit
                + _G2A * digital
            )
        
        if background:
            if ry % 2:
                datacube.data[rx,ry] -= background_even
                datacube.data[rx,ry] -= _debounce_frame(datacube.data[rx,ry])
                datacube.data[rx,ry] *= _FFB
            else:
                datacube.data[rx,ry] -= background_odd
                datacube.data[rx,ry] -= _debounce_frame(datacube.data[rx,ry])
                datacube.data[rx,ry] *= _FFA


def _load_EMPAD2_datacube(
    filepath, 
    scan_size=None, 
    background_even=None, 
    background_odd=None, 
    ):
    # get file size
    filesize = os.path.getsize(filepath)
    if scan_size is None:
        pattern_size = 128 * 128 * 4  # 4 bytes per pixel
        N_patterns = filesize / pattern_size
        Nxy = np.sqrt(N_patterns)
    
        # Check that it's reasonably square
        assert np.abs(Nxy - np.round(Nxy)) <= 1e-10, "Did you do a non-square scan?"
        Nx, Ny = Nxy, Nxy
    else:
        Nx, Ny = scan_size

    data_shape = (int(Nx), int(Ny), 128, 128)

    with open(filepath, "rb") as fid:
        datacube = py4DSTEM.DataCube(np.fromfile(fid, np.float32).reshape(data_shape))

    debounce = _process_EMPAD2_datacube(datacube, background_even, background_odd)

    return datacube




