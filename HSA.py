# from PyQt5 import QtWidgets
# from pyqtgraph import QtGui as QtWidgets
from pyqtgraph import QtGui as QtWidgets
import HSA

if __name__ == "__main__":
    qapp = QtWidgets.QApplication([])
    app = HSA.HSA()
    qapp.exec()