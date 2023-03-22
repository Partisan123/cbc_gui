import cv2
import sys
import numpy as np
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from Detection_CBC import ObjectDetection


POLYGON_ZONE = np.array([
    [140, 210],
    [650, 210],
    [650, 570],
    [140, 570]
])

#
# class Thread(QtCore.QThread):
#     changePixmap = QtCore.pyqtSignal(QtGui.QImage)
#
#     def __init__(self, parent=None):
#         QtCore.QThread.__init__(self, parent)
#         self.status = True
#         self.cap = True
#
#     def run(self):
#         res = (1024, 768)
#         # self.cap = cv2.VideoCapture('./vids/test28.avi')
#         # # # self.cap = cv2.VideoCapture(0)
#         while True:
#
#             # ret, frame = self.cap.read()
#             detect = ObjectDetection()
#
#             if True:
#                 # resized = cv2.resize(frame, res)
#                 # resized_b = resized.copy()
#                 frm = detect.det()
#
#                 rgb_img = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
#                 h, w, ch = rgb_img.shape
#                 bytesPerLine = ch * w
#                 convert2qt = QtGui.QImage(rgb_img.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
#                 scale = convert2qt.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
#                 self.changePixmap.emit(scale)
#             sys.exit(-1)


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Test App")
        self.resize(1000, 600)

        # Declarations
        self.table = QtWidgets.QTableWidget(self)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()

        # self.vid_label = QtWidgets.QLabel(self)

        # CV Thread

        # self.th = Thread(self)
        # self.th.finished.connect(self.close)
        # self.th.changePixmap.connect(self.setImage)

        # Main Layout

        main_layout = QtWidgets.QGridLayout()
        lay1_v = QtWidgets.QVBoxLayout()

        # Misc
        self.button1 = QtWidgets.QPushButton("start")
        self.button2 = QtWidgets.QPushButton("stop")

        # Create tabs

        # Tab1
        self.tab1 = QtWidgets.QWidget()
        self.tab1.layout = QtWidgets.QVBoxLayout()
        self.tab1.layout2 = QtWidgets.QHBoxLayout()

        self.tab1.layout.addWidget(self.table, 0, QtCore.Qt.AlignRight)
        self.tab1.layout.addWidget(self.button1, 0, QtCore.Qt.AlignBaseline)
        self.tab1.layout.addWidget(self.button2, 0, QtCore.Qt.AlignBaseline)
        # self.tab1.layout2.addWidget(self.vid_label, 0, QtCore.Qt.AlignLeft)

        self.tab1.layout2.addLayout(self.tab1.layout)
        self.tab1.setLayout(self.tab1.layout2)

        # Tab2
        self.tab2 = QtWidgets.QWidget()
        self.tab2.layout = QtWidgets.QVBoxLayout()
        self.tab2.layout.addWidget(QtWidgets.QTableWidget())
        self.tab2.setLayout(self.tab2.layout)

        # Tab Parent
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self.tab1, "Real Time")
        self.tabs.addTab(self.tab2, "Overall")

        # Set Main layout
        main_layout.addWidget(self.tabs, 0, 0)
        self.setLayout(main_layout)

        # Actions

        self.button1.pressed.connect(self.start)
        # self.button2.pressed.connect(self.kill_thread)
        self.button2.setEnabled(False)

        self.event_table()

    # @QtCore.pyqtSlot()
    # def kill_thread(self):
    #     print("Finishing...")
    #     self.button2.setEnabled(False)
    #     self.button1.setEnabled(True)
    #     self.th.cap.release()
    #     cv2.destroyAllWindows()
    #     self.status = False
    #     self.th.terminate()
    #     # Give time for the thread to finish
    #     time.sleep(1)
    #
    # @QtCore.pyqtSlot()
    # def start(self):
    #     print("Starting...")
    #     self.button2.setEnabled(True)
    #     self.button1.setEnabled(False)
    #     self.th.start()
    #
    # @QtCore.pyqtSlot(QtGui.QImage)
    # def setImage(self, image):
    #     self.vid_label.setPixmap(QtGui.QPixmap.fromImage(image))

    def start(self):
        print("Starting...")
        self.button2.setEnabled(True)
        self.button1.setEnabled(False)
        ObjectDetection().det()

    def event_table(self):

        self.table.setColumnCount(3)
        self.table.setRowCount(6)
        self.table.setHorizontalHeaderLabels(['Date', 'ID', 'Pass/Fail'])
        # self.table


if __name__ == "__main__":

    app = QtWidgets.QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())










