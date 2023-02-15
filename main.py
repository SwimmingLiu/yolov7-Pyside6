from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from recognizition import *
if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon('史伟铭-Portrait.ico'))
    yolo = MyWindow()
    glo._init()
    glo.set_value('yolo', yolo)
    Glo_yolo = glo.get_value('yolo')
    Glo_yolo.show()
    app.exec()
