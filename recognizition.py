import os
import shutil
import sys
import cv2
import numpy as np
from PySide6.QtGui import QPixmap, QImage, QMouseEvent, QGuiApplication
from PySide6.QtWidgets import QMessageBox, QFileDialog, QMainWindow, QWidget, QApplication
from PySide6.QtUiTools import QUiLoader, loadUiType
from PySide6.QtCore import QFile, QTimer, Qt, QEventLoop, QThread
from PySide6 import QtCore, QtGui

from PIL import Image
from lib import glo
from YoloClass import YoloThread

GLOBAL_STATE = True

formType, baseType = loadUiType("./ui/main.ui")


class YOLO(formType, baseType):
    def __init__(self):
        super().__init__()
        # 加载UI
        self.setupUi(self)
        self.setWindowFlags(Qt.CustomizeWindowHint)
        # ui部件功能设置
        self.inputPath = ""
        # Slider
        self.con_slider.valueChanged.connect(self.ValueChange)
        self.iou_slider.valueChanged.connect(self.ValueChange)
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0

        # TOOLS
        self.folder.clicked.connect(self.Selectfile)
        self.importbtn.clicked.connect(self.Import)
        self.exporter.clicked.connect(self.Export)

        # 最大化 最小化 关闭
        # 最大化按钮图片变化
        MaxIcon = QtGui.QIcon()
        MaxIcon.addPixmap(QtGui.QPixmap("./img/icons/square.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MaxIcon.addPixmap(QtGui.QPixmap("./img/icons/reduce.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        MaxIcon.addPixmap(QtGui.QPixmap("./img/icons/reduce.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.MaxButton.setCheckable(True)
        self.MaxButton.setIcon(MaxIcon)
        self.MaxButton.clicked.connect(self.max_or_restore)
        self.MinButton.clicked.connect(self.showMinimized)
        self.CloseButton.clicked.connect(self.close)
        # 视频预览
        self.input.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)
        self.output.setAlignment(QtCore.Qt.AlignCenter | QtCore.Qt.AlignVCenter)

        # 视频操作
        # 播放按钮图片变化
        PlayIcon = QtGui.QIcon()
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/pause.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/play-button.png"), QtGui.QIcon.Disabled, QtGui.QIcon.Off)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/pause.png"), QtGui.QIcon.Disabled, QtGui.QIcon.On)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/play-button.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/pause.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/play-button.png"), QtGui.QIcon.Selected, QtGui.QIcon.Off)
        PlayIcon.addPixmap(QtGui.QPixmap("./img/icons/pause.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.playbtn.setCheckable(True)
        self.playbtn.setIcon(PlayIcon)

        # 自动加载 pt文件
        self.comboBox.clear()
        self.pt_Path = "./ptmodel"
        self.pt_list = os.listdir('./ptmodel')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./ptmodel/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)
        self.comboBox.currentTextChanged.connect(self.change_model)

        # yolov7 thread
        self.yolo_thread = YoloThread()
        # 获取模型
        self.model_type = self.comboBox.currentText()
        self.yolo_thread.weights = "./ptmodel/%s" % self.model_type
        self.yolo_thread.percent_length = self.progressBar.maximum()
        self.yolo_thread.send_input.connect(lambda x: self.showimg(x, self.input, 'img'))
        self.yolo_thread.send_output.connect(lambda x: self.showimg(x, self.output, 'img'))
        self.yolo_thread.send_result.connect(self.show_result)
        self.yolo_thread.send_msg.connect(lambda x: self.foot_print(x))
        self.yolo_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        # self.yolo_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        # 运行或停止 检测
        self.playbtn.clicked.connect(self.run_or_continue)
        self.stopbtn.clicked.connect(self.stop)

    # 寻找pt模型
    def search_pt(self):
        pt_list = os.listdir('./ptmodel')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./ptmodel/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    # Conf 和 IoU 变化
    def ValueChange(self):
        self.numcon = self.con_slider.value() / 100.0
        self.numiou = self.iou_slider.value() / 100.0
        self.con_num.setValue(self.numcon)
        self.yolo_thread.conf = self.numcon
        self.iou_num.setValue(self.numiou)
        self.yolo_thread.iou = self.numiou

    # Model 变化
    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.yolo_thread.weights = "./ptmodel/%s" % self.model_type

    # 显示Label图片
    @staticmethod
    def showimg(img, label, flag):
        try:
            if flag == "path":
                img_src = cv2.imdecode(np.fromfile(img, dtype=np.uint8), -1)
            else:
                img_src = img
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw / w > ih / h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))
            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    # 选择照片/视频 并展示
    def Selectfile(self):
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要上传的图片/视频",  # 标题
            "./",  # 默认打开路径为当前路径
            "图片/视频类型 (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file == "":
            pass
        else:
            self.inputPath = file
            glo.set_value('inputPath', self.inputPath)
            if ".avi" in self.inputPath or ".mp4" in self.inputPath:
                # 显示第一帧
                self.cap = cv2.VideoCapture(self.inputPath)
                ret, frame = self.cap.read()
                if ret:
                    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.showimg(rgbImage, self.input, 'img')
            else:
                self.showimg(self.inputPath, self.input, 'path')

    # 导入模块
    def Import(self):
        file, _ = QFileDialog.getOpenFileName(
            self,  # 父窗口对象
            "选择你要导入的模型",  # 标题
            "./",  # 默认打开路径为当前路径
            "图片/视频类型 (*.pt)"  # 选择类型过滤项，过滤内容在括号中
        )
        if file == "":
            pass
        else:
            shutil.copy(file, self.pt_Path)
            QMessageBox.information(self, '提示', '模块导入成功!')

    # 导出结果
    def Export(self):
        self.OutputDir, _ = QFileDialog.getSaveFileName(
            self,  # 父窗口对象
            "导出图片/视频",  # 标题
            r".",  # 起始目录
            "图片类型 (*.jpg *.jpeg *.png *.bmp *.dib  *.jpe  *.jp2 *.mp4)"  # 选择类型过滤项，过滤内容在括号中
        )
        if self.output == "":
            QMessageBox.warning(self, '提示', '请先选择图片/视频保存的位置')
        else:
            try:
                shutil.copy(self.yolo_thread.save_path, self.OutputDir)
                QMessageBox.warning(self, '提示', '导出成功!')
            except Exception as e:
                QMessageBox.warning(self, '提示', '请先完成识别工作')
                print(e)

    # 最大化最小化窗口
    def max_or_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status:
            self.showMaximized()
            GLOBAL_STATE = False
        else:
            self.showNormal()
            GLOBAL_STATE = True

    # 开始/暂停 预测
    def run_or_continue(self):
        if self.inputPath == "":
            QMessageBox.warning(self, "提示", "请先选择需要识别的图片/视频!")
        else:
            self.yolo_thread.jump_out = False
            if self.playbtn.isChecked():
                self.yolo_thread.is_continue = True
                if not self.yolo_thread.isRunning():
                    self.yolo_thread.start()
                    self.foot_print("开始检测>>>>>" + self.inputPath )
            else:
                self.yolo_thread.is_continue = False
                self.foot_print('Pause')

    # 停止识别
    def stop(self):
        self.yolo_thread.jump_out = True
        self.foot_print('Stop')

    # 统计结果
    def show_result(self, statistic_dic):
        try:
            self.resultlist.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' ' + str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultlist.addItems(results)

        except Exception as e:
            print(repr(e))

    # foot栏 输出结果
    def foot_print(self, msg):
        if msg in ['Stop','Finished']:
            self.playbtn.setChecked(False)
        self.outputbox.setText(msg)


class MyWindow(YOLO):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.center()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.mouse_start_pt = event.globalPosition().toPoint()
            self.window_pos = self.frameGeometry().topLeft()
            self.drag = True

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.drag:
            distance = event.globalPosition().toPoint() - self.mouse_start_pt
            self.move(self.window_pos + distance)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.drag = False

    def center(self):
        # PyQt6获取屏幕参数
        screen = QGuiApplication.primaryScreen().size()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2,
                  (screen.height() - size.height()) / 2 - 10)
