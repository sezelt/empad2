import py4DSTEM
import numpy as np
import os
from pathlib import Path
import h5py
from typing import Callable, Optional
from typing_extensions import TypedDict

__all__ = ["load_calibration_data", "load_background", "load_dataset"]

CalibrationSet = TypedDict(
    "CalibrationSet", {"data": dict[str, np.ndarray], "method": Callable}
)


def load_calibration_data(
    sensor: Optional[str] = None,
    filepath: Optional[str | Path] = None,
    method: Optional[str] = None,
) -> CalibrationSet:
    """
    Import calibration data for the sensor. Can be called in two ways:

    If using calibration files included with the package, use the name of the microscope:
    calibrations = empad2.load_calibration_data(sensor="cryo-titan")
    calibrations = empad2.load_calibration_data(sensor="andromeda")

    If using your own calibration files, pass the path to the calibration data
    as well as the type of model.
    calibrations = empad2.load_calibration_data(filepath="/path/to/data.h5", method="linear")
    Model can be "linear" or "quadratic".
    "linear" uses G1A, G1B, G2A, G2B, B2A, B2B, FFA, FFB
    "quadratic" uses Ml, Md, alpha, Oh, Ot, FFA, FFB
    """

    processing_methods = {
        "linear": _process_EMPAD2_datacube_linear,
        "quadratic": _process_EMPAD2_datacube_quadratic,
    }

    if sensor is not None:
        # load a bundled sensor
        sensor_name = sensor.lower()
        if sensor_name == "cryo-titan":
            with h5py.File(
                Path(__file__).resolve().parent / "calibration_data" / "cryo-titan-calibrations.h5"
            ) as cal_file:
                data = {
                    "G1A": cal_file["G1A"][()],
                    "G2A": cal_file["G2A"][()],
                    "G1B": cal_file["G1B"][()],
                    "G2B": cal_file["G2B"][()],
                    "B2A": cal_file["B2A"][()],
                    "B2B": cal_file["B2B"][()],
                    "FFA": cal_file["FFA"][()],
                    "FFB": cal_file["FFB"][()],
                }
                cal_method = processing_methods["linear"]
        elif sensor_name == "andromeda":
            with h5py.File(
                Path(__file__).resolve().parent / "calibration_data" / "andromeda-calibrations.h5"
            ) as cal_file:
                data = {
                    "Ml": cal_file["Ml"][()],
                    "alpha": cal_file["alpha"][()],
                    "Md": cal_file["Md"][()],
                    "Oh": cal_file["Oh"][()],
                    "Ot": cal_file["Ot"][()],
                    "FFA": cal_file["FFA"][()],
                    "FFB": cal_file["FFB"][()],
                }
                cal_method = processing_methods["quadratic"]

        else:
            raise ValueError("Sensor name not recognized")
    elif filepath is not None and method is not None:
        # read any arrays in the file
        with h5py.File(filepath) as cal_file:
            data = {k: cal_file[k][()] for k in cal_file.keys()}

        cal_method = processing_methods[method]

    else:
        raise ValueError("Either sensor name or path and method must be specified.")

    return {"data": data, "method": cal_method}


def load_background(filepath, calibration_data: dict, scan_size=None):
    bg_data = _load_EMPAD2_datacube(
        filepath, calibration_data=calibration_data, scan_size=scan_size
    )
    background_odd = np.mean(bg_data.data[:, ::2], axis=(0, 1))
    background_even = np.mean(bg_data.data[:, 1::2], axis=(0, 1))
    return {"even": background_even, "odd": background_odd}


def load_dataset(
    filepath: str,
    background: dict,
    calibration_data: CalibrationSet,
    scan_size=None,
):
    return _load_EMPAD2_datacube(
        filepath,
        calibration_data=calibration_data,
        scan_size=scan_size,
        background_even=background["even"],
        background_odd=background["odd"],
    )


##############################################################

# Constant data used in debouncing
binwidth = 10
histogram_bins, binwidth = np.linspace(
    -200 - binwidth // 2, 220 - binwidth // 2, num=420 // binwidth + 1, retstep=True
)
# get coordinates for doing polynomial fit
bin_centers = (histogram_bins[:-1] + histogram_bins[1:]) / 2.0
bin_centers_sq = bin_centers**2
fit_coords = np.stack((np.ones_like(bin_centers), bin_centers, bin_centers_sq), axis=1)


def _debounce_frame(frame, fit_range=5):
    hist, _ = np.histogram(frame.flat, bins=histogram_bins)
    hist_peak = np.argmax(hist)

    if hist_peak < fit_range or hist_peak > len(hist) - fit_range - 1:
        return 0.0

    polyfit = np.linalg.lstsq(
        fit_coords[hist_peak - fit_range : hist_peak + fit_range],
        hist[hist_peak - fit_range : hist_peak + fit_range],
        rcond=None,
    )[0]

    polymax = -polyfit[1] / 2 / polyfit[2]

    # make sure the peak is inside the fit range, else return 0.0
    return (
        polymax
        if (
            polymax > bin_centers[hist_peak - fit_range]
            and polymax < bin_centers[hist_peak + fit_range]
        )
        else 0.0
    )


def _process_EMPAD2_datacube_linear(
    datacube: py4DSTEM.DataCube,
    calibration_data: CalibrationSet,
    background_even=None,
    background_odd=None,
) -> None:
    # get calibration data from file
    _G1A = calibration_data["data"]["G1A"]
    _G2A = calibration_data["data"]["G2A"]
    _G1B = calibration_data["data"]["G1B"]
    _G2B = calibration_data["data"]["G2B"]
    _B2A = calibration_data["data"]["B2A"]
    _B2B = calibration_data["data"]["B2B"]
    _FFA = calibration_data["data"]["FFA"]
    _FFB = calibration_data["data"]["FFB"]

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
                datacube.data[rx, ry] -= background_even
                datacube.data[rx, ry] -= _debounce_frame(datacube.data[rx, ry])
                datacube.data[rx, ry] *= _FFB
            else:
                datacube.data[rx, ry] -= background_odd
                datacube.data[rx, ry] -= _debounce_frame(datacube.data[rx, ry])
                datacube.data[rx, ry] *= _FFA


def _process_EMPAD2_datacube_quadratic(
    datacube: py4DSTEM.DataCube,
    calibration_data: CalibrationSet,
    background_even=None,
    background_odd=None,
) -> None:
    Ml = calibration_data["data"]["Ml"]
    alpha = calibration_data["data"]["alpha"]
    Md = calibration_data["data"]["Md"]
    Oh = calibration_data["data"]["Oh"]
    Ot = calibration_data["data"]["Ot"]
    _FFA = calibration_data["data"]["FFA"]
    _FFB = calibration_data["data"]["FFB"]

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

        datacube.data[rx, ry] = (
            analog * (1 - gain_bit)  # analog part
            + Ml[:, :, ry % 2] * analog * gain_bit  # ml
            + alpha[:, :, ry % 2] * (analog * gain_bit) ** 2  # alpha
            + Md[:, :, ry % 2] * digital  # md
            + Oh[:, :, ry % 2] * gain_bit  # oh
            - Ot[:, :, ry % 2]  # ot
        )

        if background:
            if ry % 2:
                datacube.data[rx, ry] -= background_even
                datacube.data[rx, ry] -= _debounce_frame(datacube.data[rx, ry])
                datacube.data[rx, ry] *= _FFB
            else:
                datacube.data[rx, ry] -= background_odd
                datacube.data[rx, ry] -= _debounce_frame(datacube.data[rx, ry])
                datacube.data[rx, ry] *= _FFA


def _load_EMPAD2_datacube(
    filepath,
    calibration_data: CalibrationSet,
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

    # Call the correct calibration method, as determined by the calibration dict
    calibration_data["method"](
        datacube, calibration_data, background_even, background_odd
    )

    return datacube
