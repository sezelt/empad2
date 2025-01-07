from functools import partial
import numpy as np

from PyQt5.QtWidgets import QWidget, QActionGroup, QMenu, QFileDialog, QMessageBox

import empad2
from py4D_browser.utils import StatusBarWriter


class EMPAD2Plugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "sezelt.empad2.converter"

    # optional flags

    # Plugins may add a top-level menu on their own, or can opt to have
    # a submenu located under Plugins>[display_name], which is created before
    # initialization and its QMenu object passed as `plugin_menu`
    uses_plugin_menu = False
    display_name = "EMPAD2 Reader"

    def __init__(self, parent, *args, **kwargs):
        super().__init__()
        self.parent = parent

        self.empad2_calibrations = None
        self.empad2_background = None

        self.empad2_menu = QMenu("&EMPAD-G2", self)
        # For historical reasons, keep the EMPAD2 menu in second position
        parent.menu_bar.insertMenu(parent.menu_bar.actions()[1], self.empad2_menu)

        sensor_menu = self.empad2_menu.addMenu("&Sensor")
        calibration_action_group = QActionGroup(self)
        calibration_action_group.setExclusive(True)

        for name, sensor in empad2.SENSORS.items():
            menu_item = sensor_menu.addAction(sensor["display-name"])
            calibration_action_group.addAction(menu_item)
            menu_item.setCheckable(True)
            menu_item.triggered.connect(partial(self.set_empad2_sensor, name))

        self.empad2_menu.addAction("Load &Background...").triggered.connect(
            self.load_empad2_background
        )
        self.empad2_menu.addAction("Load &Dataset...").triggered.connect(
            self.load_empad2_dataset
        )

    def close(self):
        pass  # perform any shutdown activities

    def set_empad2_sensor(self, sensor_name):
        self.empad2_calibrations = empad2.load_calibration_data(sensor=sensor_name)
        self.parent.statusBar().showMessage(f"{sensor_name} calibrations loaded", 5_000)

    def load_empad2_background(self):
        if self.empad2_calibrations is not None:
            filename = raw_file_dialog(self.parent)
            self.empad2_background = empad2.load_background(
                filepath=filename, calibration_data=self.empad2_calibrations
            )
            self.parent.statusBar().showMessage("Background data loaded", 5_000)
        else:
            QMessageBox.warning(
                self.parent, "No calibrations loaded!", "Please select a sensor first"
            )

    def load_empad2_dataset(self):
        if self.empad2_calibrations is not None:
            dummy_data = False
            if self.empad2_background is None:
                continue_wo_bkg = QMessageBox.question(
                    self.parent,
                    "Load without background?",
                    "Background data has not been loaded. Do you want to continue loading data?",
                )
                if continue_wo_bkg == QMessageBox.No:
                    return
                else:
                    self.empad2_background = {
                        "even": np.zeros((128, 128), dtype=np.float32),
                        "odd": np.zeros((128, 128), dtype=np.float32),
                    }
                    dummy_data = True

            filename = raw_file_dialog(self.parent)
            self.parent.datacube = empad2.load_dataset(
                filename,
                self.empad2_background,
                self.empad2_calibrations,
                _tqdm_args={
                    "desc": "Loading",
                    "file": StatusBarWriter(self.parent.statusBar()),
                    "mininterval": 1.0,
                },
            )

            if dummy_data:
                self.empad2_background = None

            self.parent.update_diffraction_space_view(reset=True)
            self.parent.update_real_space_view(reset=True)

            self.parent.setWindowTitle(filename)

        else:
            QMessageBox.warning(
                self.parent, "No calibrations loaded!", "Please select a sensor first"
            )


def raw_file_dialog(browser):
    filename = QFileDialog.getOpenFileName(
        browser,
        "Open EMPAD-G2 Data",
        "",
        "EMPAD-G2 Data (*.raw);;Any file(*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")
