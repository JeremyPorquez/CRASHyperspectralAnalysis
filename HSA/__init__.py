try : from .HSAGUI.main import Ui_MainWindow
except : from HSAGUI.main import Ui_MainWindow
from . import pyqtgraph as pg
from .pyqtgraph import QtCore
from .pyqtgraph import QtGui as QtWidgets
# from pyqtgraph import QtGui as QtWidgets
# from pyqtgraph import QtCore

from . import tiff
from . import svd
from . import ramancsv
from . import mplcanvas
from . import CARS
from multiprocessing.pool import ThreadPool
import os
import numpy as np
import pandas as pd


class HSA:
    class Signal(QtCore.QObject):
        image_loaded = QtCore.pyqtSignal()
        applying_ansc_transform = QtCore.pyqtSignal()
        applied_ansc_transform = QtCore.pyqtSignal()
        setting_ansc_transform = QtCore.pyqtSignal()
        set_ansc_transform = QtCore.pyqtSignal()


    def __init__(self):
        self.signal = self.Signal()
        self.createUi()
        self._reinit()


    def _reinit(self):
        self.data = None
        self.raman_index = None
        self.nrb_bg_intensity = None
        self.retrieved = None
        self.new_image_loaded = False

    def createUi(self):
        self.mainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.mainWindow)
        self.setupStatusSignals()
        self.createPgItems()
        self.createMplItems()
        self.setupButtons()
        self.ui.tabWidget.setCurrentIndex(0)
        self.mainWindow.show()

    def setupStatusSignals(self):
        def info(message, timeout=0):
            self.ui.statusbar.showMessage(message, timeout)
        self.signal.applying_ansc_transform.connect(lambda: info('Applying Anscombe-SVD filter'))
        self.signal.setting_ansc_transform.connect(lambda: info('Setting Anscombe-SVD filter value'))
        self.signal.set_ansc_transform.connect(lambda: info('Anscombe-SVD filter value set'))
        self.signal.applied_ansc_transform.connect(lambda: info('Anscombe-SVD filter applied'))
        self.signal.image_loaded.connect(lambda: info('Image Loaded'))
        info('Hyperspectral Image c/o JGPorquez')

    def createPgItems(self):
        self.image_tiff = pg.ImageView()
        self.image_svd = pg.ImageView()
        self.ui.pglayout.addWidget(self.image_tiff)
        self.ui.svdLayout.addWidget(self.image_svd)
        self.image_tiff.timeLine.sigPositionChanged.connect(
            lambda: self.update_pgimage_position(self.image_tiff,
                                                 self.ui.tiff_position_doubleSpinBox))
        self.image_svd.timeLine.sigPositionChanged.connect(
            lambda: self.update_pgimage_position(self.image_svd,
                                                 self.ui.svd_position_doubleSpinBox))

    def createMplItems(self):
        self.mplPlot = mplcanvas.MplCanvas(self.mainWindow)
        self.mplPlot.createPlot()
        self.ui.ramanRetrievalLayout.addWidget(self.mplPlot)
        self.navi_toolbar = mplcanvas.NavigationToolbar(self.mplPlot, self.mainWindow)
        self.ui.ramanRetrievalLayout.addWidget(self.navi_toolbar)

    def setupButtons(self):
        self.ui.openTiff.clicked.connect(self.open_tiff)
        self.ui.saveTiffROI.clicked.connect(lambda: self.save_roi(self.image_tiff))
        self.ui.setTiffROItoCARS.clicked.connect(lambda: self.set_roi_as_cars(self.image_tiff))
        self.ui.setTiffROItoBG.clicked.connect(lambda: self.set_roi_as_background(self.image_tiff))
        self.ui.openWN.clicked.connect(lambda: self.open_wn(None))
        self.ui.applySVD.clicked.connect(self.apply_svd)
        self.ui.saveSVD.clicked.connect(self.save_svd)
        self.ui.saveSVD_all.clicked.connect(self.save_svd_all)
        self.ui.saveSVDROI.clicked.connect(lambda: self.save_roi(self.image_svd))
        self.ui.setSVDValue.clicked.connect(lambda: self.set_svd_value())
        self.ui.setSVDROItoCARS.clicked.connect(lambda: self.set_roi_as_cars(self.image_svd))
        self.ui.setSVDROItoBG.clicked.connect(lambda: self.set_roi_as_background(self.image_svd))
        self.ui.openBackground.clicked.connect(lambda: self.open_background(None))
        self.ui.openCARSIntensity.clicked.connect(lambda: self.open_cars(None))
        self.ui.applyRetrieval.clicked.connect(self.apply_retrieval)
        self.ui.saveRetrieved.clicked.connect(self.save_retrieved)
        self.ui.tiff_position_doubleSpinBox.valueChanged.connect(
            lambda: self.set_pgimage_position(self.image_tiff,
                                              self.ui.tiff_position_doubleSpinBox))
        self.ui.svd_position_doubleSpinBox.valueChanged.connect(
            lambda: self.set_pgimage_position(self.image_svd,
                                              self.ui.svd_position_doubleSpinBox))


    def loadFiles(self):
        idx = 0
        for file in self.filenames:
            fname, ext = os.path.splitext(file)
            if any(x in ext for x in ('tiff','tif')):
                self.filename_tiff = file
                data = tiff.imread(file)
                self.data = svd.Image(data)
                idx += 1
            if any(x in ext for x in ('csv')):
                self.open_wn(file)

        return self.data


    def open_tiff(self):
        fileDialog = QtWidgets.QFileDialog()
        fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        filter = "TIFF (*.tiff);;TIF (*.tif)"
        defaultDirectory = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        files, filter = fileDialog.getOpenFileNames(QtWidgets.QWidget(), "Open files")
        self.filenames = files

        if len(self.filenames) > 0:
            # self._reinit()
            self.loadFiles()
            self.update_pgimage(self.image_tiff,self.data.raw_image)
            z,y,x = self.data.shape
            bitsize = self.data.dtype.name
            image_info_text = "{} {}x{}x{}".format(bitsize,z,x,y)
            self.ui.image_info_label.setText(image_info_text)


    def open_wn(self, file=None):
        if file is None:
            fileDialog = QtWidgets.QFileDialog()
            fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            # defaultDirectory = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
            file, filter = fileDialog.getOpenFileName(QtWidgets.QWidget(), "Open file")
        if file == '':
            return None
        wn_dataframe = pd.read_csv(file)
        self.raman_index = ramancsv.getRamanIndex(wn_dataframe)
        self.update_pgimage(self.image_tiff,self.data.raw_image)
        self.update_pgimage(self.image_svd,self.data.svd_image)

    def open_background(self,file=None,col=1):
        if file is None:
            fileDialog = QtWidgets.QFileDialog()
            fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            file, filter = fileDialog.getOpenFileName(QtWidgets.QWidget(),"Open file")
            fname, ext = os.path.splitext(file)
            if fname == '':
                return None
            if 'csv' in ext:
                background = pd.read_csv(file)
            if any(x in ext for x in ('xls', 'xlsx')):
                background = pd.read_excel(file)
            if 'Y' in background.columns:
                nrb_bg_intensity = background.Y
            else:
                nrb_bg_intensity = background[background.columns[col]].values
            if 'Raman' in  background.columns:
                index = background.Raman.values
            elif 'X' in  background.columns:
                index = background.X.values
            else:
                index = background.index
            self.raman_index = index
            self.nrb_bg_intensity = nrb_bg_intensity
            self.plot_background()

    def open_cars(self,file=None,col=1):
        if file is None:
            fileDialog = QtWidgets.QFileDialog()
            fileDialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
            file, filter = fileDialog.getOpenFileName(QtWidgets.QWidget(),"Open file")
            fname, ext = os.path.splitext(file)
            if fname == '':
                return None
            if 'csv' in ext:
                cars = pd.read_csv(file)
            if any(x in ext for x in ('xls','xlsx')):
                cars = pd.read_excel(file)
            if 'Y' in cars.columns:
                cars_intensity= cars.Y.values
            else:
                cars_intensity = cars[cars.columns[col]].values
            if 'Raman' in cars.columns:
                index = cars.Raman.values
            elif 'X' in cars.columns:
                index = cars.X.values
            else:
                index = cars.index
            self.raman_index = index
            self.cars_intensity = cars_intensity
            self.plot_cars()

    def plot_background(self):
        self.mplPlot.plots[0].set_data(self.raman_index, self.nrb_bg_intensity)
        self.mplPlot.axes[0].set_xlim(self.raman_index.min(), self.raman_index.max())
        self.mplPlot.axes[0].set_ylim(self.nrb_bg_intensity.min(), self.nrb_bg_intensity.max())
        self.mplPlot.draw()

    def plot_cars(self):
        self.mplPlot.plots[1].set_data(self.raman_index, self.cars_intensity)
        self.mplPlot.axes[1].set_xlim(self.raman_index.min(), self.raman_index.max())
        self.mplPlot.axes[1].set_ylim(self.cars_intensity.min(), self.cars_intensity.max())
        self.mplPlot.draw()

    def plot_retrieved(self):
        self.mplPlot.plots[2].set_data(self.raman_index, self.retrieved)
        self.mplPlot.axes[2].set_xlim(self.raman_index.min(), self.raman_index.max())
        self.mplPlot.axes[2].set_ylim(self.retrieved.min(), self.retrieved.max())
        self.mplPlot.draw()

    def apply_svd(self):
        if self.data.raw_image is not None:
            self.signal.applying_ansc_transform.emit()
            singular_values = self.ui.singularValues_spinBox.value()
            self.data.calc_svd(singular_values=singular_values)
            self.update_SVDPgImage()


    def apply_retrieval(self):
        if (self.nrb_bg_intensity is None) & (self.cars_intensity is None):
            return None
        if len(self.nrb_bg_intensity) != len(self.cars_intensity):
            print('NRB and CARS have different shapes')
            return None

        smoothness_exponent = self.ui.smoothness_spinbox.value()
        smoothness = 10**smoothness_exponent
        asymmetry_exponent = self.ui.asymmetry_spinbox.value()
        asymmetry = 10**asymmetry_exponent
        savgol_window = self.ui.savgol_window_retr_spinbox.value()
        try :
            self.retrieved = CARS.getCorrectedCARSPhase(I_CARS=self.cars_intensity,
                                                        I_REF=self.nrb_bg_intensity,
                                                        SMOOTHNESS_PARAM=smoothness,
                                                        ASYM_PARAM=asymmetry,
                                                        SAVGOL_WINDOW=savgol_window)
            self.plot_retrieved()
        except Exception as e:
            print(e)

    def apply_img_retrieval(self):
        if (self.nrb_bg_intensity is None) & (self.data.raw_image is None):
            return None
        # if len(self.nrb_bg_intensity) != len(self.cars_intensity):
        #     print('NRB and CARS have different shape')
        #     return None
        smoothness_exponent = self.ui.smoothness_spinbox.value()
        smoothness = 10**smoothness_exponent
        asymmetry_exponent = self.ui.asymmetry_spinbox.value()
        asymmetry = 10**asymmetry_exponent
        savgol_window = self.ui.savgol_window_retr_spinbox.value()

        img = self.data.image

        self.retrieved_image = CARS.getCorrectedCARSPhaseImage(img,
                                                               I_REF=self.nrb_bg_intensity,
                                                               SMOOTHNESS_PARAM=smoothness,
                                                               ASYM_PARAM=asymmetry,
                                                               SAVGOL_WINDOW=savgol_window)
        # self.update_pgimage()


    def save_roi(self,imageView):
        assert isinstance(imageView, pg.ImageView)
        if len(imageView.roiCurves) == 0:
            return None
        fileDialog = QtWidgets.QFileDialog()
        filter = "CSV (*.csv)"
        file, filt = fileDialog.getSaveFileName(QtWidgets.QWidget(), "Save CSV", filter=filter)
        roiCurve = imageView.roiCurves[0]
        x,y = roiCurve.xData, roiCurve.yData
        try :
            df = pd.DataFrame(y,index=x, columns=['Y'])
            df.index.name = 'X'
            df.to_csv(file)
        except Exception as e:
            print('Error in saving ROI : {}'.format(e))


    def save_svd(self):
        if self.data.svd_image is not None:
            filter = "TIF (*.tif)"
            fileDialog = QtWidgets.QFileDialog()
            file, filter = fileDialog.getSaveFileName(QtWidgets.QWidget(), "Save svd tiff", filter=filter)
            tiff.imsave(file,self.data.svd_image)

    def save_svd_all(self):
        if self.data.svd_image is not None:
            fileDialog = QtWidgets.QFileDialog()
            saveDirectory = fileDialog.getExistingDirectory()
            singular_values = self.ui.singularValues_spinBox.value()

            def save_sv():
                for sv in range(-1, singular_values + 1):
                    print('Saving singular value : {}'.format(sv))
                    self.data_svd.calc_svd_single(sv)
                    image = self.data.svd_image_single
                    if sv == -1:
                        filename = 'svd_full.tif'
                    else:
                        filename = 'svd_{0:0>3}.tif'.format(sv)
                    filename = os.path.join(saveDirectory,filename)
                    tiff.imsave(filename,image)
            pool = ThreadPool(processes=1)
            pool.apply_async(save_sv)

    def save_retrieved(self):
        if self.retrieved is None:
            return None

        fileDialog = QtWidgets.QFileDialog()
        filter = "CSV (*.csv)"
        file, filt = fileDialog.getSaveFileName(QtWidgets.QWidget(), "Save CSV", filter=filter)
        try :
            df = pd.DataFrame(self.retrieved, index=self.raman_index, columns=['Y'])
            df.index.name = 'X'
            df.to_csv(file)
        except Exception as e:
            print('Error in saving ROI : {}'.format(e))

    def set_roi_as_cars(self,imageView):
        assert isinstance(imageView, pg.ImageView)
        if len(imageView.roiCurves) == 0:
            return None
        roiCurve = imageView.roiCurves[0]
        x,y = roiCurve.xData, roiCurve.yData
        self.raman_index = x
        self.cars_intensity = y
        self.plot_cars()

    def set_roi_as_background(self,imageView):
        assert isinstance(imageView, pg.ImageView)
        if len(imageView.roiCurves) == 0:
            return None
        roiCurve = imageView.roiCurves[0]
        x,y = roiCurve.xData, roiCurve.yData
        self.raman_index = x
        self.nrb_bg_intensity = y
        self.plot_background()


    def set_svd_value(self, singular_value=None, updateImage=True):
        if self.data is None:
            return None
        if singular_value is None:
            singular_value = self.ui.singularValue_spinBox.value()
        self.signal.setting_ansc_transform.emit()
        self.data.calc_svd_single(singular_value)
        if updateImage:
            self.update_SVDPgImage(self.data.svd_image_single)
        self.signal.set_ansc_transform.emit()

    def update_pgimage(self,imageView,data):
        assert isinstance(imageView, pg.ImageView)
        if data is not None:
            assert isinstance(data, np.ndarray)
            raman_index = None
            if self.raman_index is not None:
                if data.shape[0] == len(self.raman_index):
                    raman_index = self.raman_index
            imageView.setImage(np.swapaxes(data,1,2),
                                     xvals=raman_index,
                                     autoLevels=True
                                     )
            imageView.autoRange()
            self.signal.image_loaded.emit()

    def update_TIFFPgImage(self):
        if self.data.raw_image is not None:
            data_tiff = self.data.raw_image.astype(np.float)
            raman_index = None
            if self.raman_index is not None:
                if data_tiff.shape[0] == len(self.raman_index):
                    raman_index = self.raman_index
            self.image_tiff.setImage(np.swapaxes(data_tiff,1,2),
                                     xvals=raman_index,
                                     autoLevels=True
                                     )
            self.image_tiff.autoRange()
            self.signal.image_loaded.emit()

    def update_SVDPgImage(self,image=None):
        if self.data.svd_image is not None:
            if image is None:
                image = self.data.svd_image
            raman_index = None
            if self.raman_index is not None:
                if image.shape[0] == len(self.raman_index):
                    raman_index = self.raman_index
            self.image_svd.setImage(np.swapaxes(image, 1, 2),
                                    xvals=raman_index,
                                    autoLevels=True
                                    )
            self.image_svd.autoRange()
            self.signal.applied_ansc_transform.emit()

    def set_pgimage_position(self, imageView, doubleSpinBox):
        if (not isinstance(imageView, pg.ImageView)) & (not isinstance(doubleSpinBox, QtWidgets.QDoubleSpinBox)):
            return None

        new_value = doubleSpinBox.value()
        current_index = imageView.currentIndex
        new_index = np.argmin(np.abs(new_value - imageView.tVals))
        current_value = np.round(imageView.tVals[current_index], 2)
        if current_index == new_index:
            if new_value > current_value:
                new_index += 1
            elif new_value < current_value:
                new_index -= 1
        try:
            imageView.setCurrentIndex(new_index)
        except Exception as e:
            print(e)

    def update_pgimage_position(self, imageview, doubleSpinBox):
        if (not isinstance(imageview, pg.ImageView)) & (not isinstance(doubleSpinBox, QtWidgets.QDoubleSpinBox)):
            return None
        value = imageview.timeLine.value()
        doubleSpinBox.setValue(value)



