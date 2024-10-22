import py4DSTEM
import numpy as np
import os
from pathlib import Path
import h5py
from typing import Callable, Optional, TypedDict
from empad2.combine import combine_quadratic, combine_quadratic_bgsub_debounce
from time import time

__all__ = ["load_calibration_data", "load_background", "load_dataset", "SENSORS"]

CalibrationSet = TypedDict(
    "CalibrationSet",
    {
        "data": dict[str, np.ndarray],
        "method": Callable,
        "postprocess": Optional[Callable],
    },
)
BackgroundSet = TypedDict("BackgroundSet", {"even": np.ndarray, "odd": np.ndarray})

SENSORS = {
    "cryo-titan": {
        "display-name": "Cryo Titan",
        "data-path": Path(__file__).resolve().parent
        / "calibration_data"
        / "cryo-titan-calibrations.h5",
        "method": "linear",
        "dataset-names": ["G1A", "G2A", "G1B", "G2B", "B2A", "B2B", "FFA", "FFB"],
    },
    "andromeda": {
        "display-name": "Andromeda",
        "data-path": Path(__file__).resolve().parent
        / "calibration_data"
        / "andromeda-calibrations.h5",
        "method": "quadratic",
        "post-process": "andromeda_dead_pixel",
        "dataset-names": ["Ml", "alpha", "Md", "Ot", "Oh", "FFA", "FFB"],
    },
}


def load_calibration_data(
    sensor: Optional[str] = None,
    filepath: Optional[str | Path] = None,
    method: Optional[str] = None,
    postprocess_method: Optional[str] = None,
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

    _constant_names = {
        "linear": ["G1A", "G2A", "G1B", "G2B", "B2A", "B2B", "FFA", "FFB"],
        "quadratic": ["Ml", "alpha", "Md", "Ot", "Oh", "FFA", "FFB"],
    }

    postprocess_methods = {
        "andromeda_dead_pixel": _andromeda_heal_dead_pixel,
    }

    if sensor is not None:
        # load a bundled sensor
        sensor_name = sensor.lower()
        if sensor_name in SENSORS.keys():
            sensor_data = SENSORS[sensor_name]
            with h5py.File(sensor_data["data-path"]) as cal_file:
                data = {k: np.array(cal_file[k]) for k in sensor_data["dataset-names"]}
            cal_method = processing_methods[sensor_data["method"]]

            postprocess = (
                postprocess_methods.get(sensor_data["post-process"])
                if "post-process" in sensor_data
                else None
            )

        else:
            raise ValueError(
                f"Sensor name not recognized. Bundled sensors are {[s['display-name'] for s in SENSORS.values()]}"
            )
    elif filepath is not None and method is not None:
        # read any arrays in the file
        with h5py.File(filepath) as cal_file:
            data = {k: np.array(cal_file[k]) for k in _constant_names[method]}

        cal_method = processing_methods[method]
        postprocess = (
            postprocess_methods.get(postprocess_method) if postprocess_method else None
        )

    else:
        raise ValueError("Either sensor name or path and method must be specified.")

    return {"data": data, "method": cal_method, "postprocess": postprocess}


def load_background(
    filepath,
    calibration_data: CalibrationSet,
    scan_size=None,
    combination_kwargs={},
) -> BackgroundSet:
    bg_data = _load_EMPAD2_datacube(
        filepath,
        calibration_data=calibration_data,
        scan_size=scan_size,
        combination_kwargs=combination_kwargs,
    )
    background_odd = np.mean(bg_data.data[:, ::2], axis=(0, 1))
    background_even = np.mean(bg_data.data[:, 1::2], axis=(0, 1))
    return {"even": background_even, "odd": background_odd}


def load_dataset(
    filepath: str,
    background: BackgroundSet,
    calibration_data: CalibrationSet,
    scan_size=None,
    _tqdm_args={},
    combination_kwargs={},
) -> py4DSTEM.DataCube:
    return _load_EMPAD2_datacube(
        filepath,
        calibration_data=calibration_data,
        scan_size=scan_size,
        background_even=background["even"],
        background_odd=background["odd"],
        _tqdm_args=_tqdm_args,
        combination_kwargs=combination_kwargs,
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
    _tqdm_args={},
    combination_kwargs={},
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
    for rx, ry in py4DSTEM.tqdmnd(
        datacube.data.shape[0], datacube.data.shape[1], **_tqdm_args
    ):
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
    background_even: Optional[np.ndarray] = None,
    background_odd: Optional[np.ndarray] = None,
    _tqdm_args={},
    combination_kwargs={},
) -> None:

    # calibration data are stored in the wrong order, should fix at the source...
    Ml = np.stack(
        [
            calibration_data["data"]["Ml"][:, :, 0],
            calibration_data["data"]["Ml"][:, :, 1],
        ]
    )
    alpha = np.stack(
        [
            calibration_data["data"]["alpha"][:, :, 0],
            calibration_data["data"]["alpha"][:, :, 1],
        ]
    )
    Md = np.stack(
        [
            calibration_data["data"]["Md"][:, :, 0],
            calibration_data["data"]["Md"][:, :, 1],
        ]
    )
    Oh = np.stack(
        [
            calibration_data["data"]["Oh"][:, :, 0],
            calibration_data["data"]["Oh"][:, :, 1],
        ]
    )
    Ot = np.stack(
        [
            calibration_data["data"]["Ot"][:, :, 0],
            calibration_data["data"]["Ot"][:, :, 1],
        ]
    )
    FF = np.stack(
        [calibration_data["data"]["FFA"][:, :], calibration_data["data"]["FFB"][:, :]]
    )

    # If backgrounds are provided, do debounce and flatfield:
    if background_even is not None and background_odd is not None:
        bkg = np.stack([background_odd, background_even])

        t0 = time()
        debounce = combine_quadratic_bgsub_debounce(
            datacube.data,
            Ml,
            alpha,
            Md,
            Oh,
            Ot,
            FF,
            bkg,
            **combination_kwargs,
        )

        print(f"Combination + debounce: {np.prod(datacube.Rshape)/(time()-t0):.0f} fps")

        debounce = py4DSTEM.VirtualImage(debounce, name="Debounce correction")
        datacube.attach(debounce)
    # Otherwise, only do binary twiddling
    else:
        t0 = time()
        combine_quadratic(
            datacube.data,
            Ml,
            alpha,
            Md,
            Oh,
            Ot,
            **combination_kwargs,
        )
        print(f"Combination: {np.prod(datacube.Rshape)/(time()-t0):.0f} fps")


def _andromeda_heal_dead_pixel(dataset: py4DSTEM.DataCube):
    """
    Heal the dead pixel at 86,90 on the Andromeda sensor
    """
    dataset.data[:, :, 85, 90] /= 2.0
    dataset.data[:, :, 86, 90] = dataset.data[:, :, 85, 90]


def _load_EMPAD2_datacube(
    filepath,
    calibration_data: CalibrationSet,
    scan_size=None,
    background_even=None,
    background_odd=None,
    _tqdm_args={},
    combination_kwargs={},
):
    # get file size
    filesize = os.path.getsize(filepath)
    if scan_size is None:
        pattern_size = 128 * 128 * 4  # 4 bytes per pixel
        N_patterns = filesize / pattern_size
        Nxy = np.sqrt(N_patterns)

        # Check that it's reasonably square
        if np.abs(Nxy - np.round(Nxy)) <= 1e-10:
            Nx, Ny = Nxy, Nxy
        else:
            Nx, Ny = 1, N_patterns
    else:
        Nx, Ny = scan_size

    data_shape = (int(Nx), int(Ny), 128, 128)

    with open(filepath, "rb") as fid:
        datacube = py4DSTEM.DataCube(np.fromfile(fid, np.float32).reshape(data_shape))

    # Call the correct calibration method, as determined by the calibration dict
    calibration_data["method"](
        datacube,
        calibration_data,
        background_even,
        background_odd,
        _tqdm_args,
        combination_kwargs,
    )

    # only postprocess if this is being loaded with background subtraction:
    if (
        background_even is not None
        and background_odd is not None
        and calibration_data["postprocess"]
    ):
        calibration_data["postprocess"](datacube)

    return datacube
