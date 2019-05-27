import sys
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLineEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QLabel, \
    QTextEdit, QFormLayout, QPushButton, QRadioButton, QButtonGroup, QFileDialog, QProgressBar, QComboBox, QSpinBox
from PyQt5.QtCore import Qt, QTimer, QBasicTimer, QDateTime
import time
import datetime

# class Example
from qtpy import QtGui
from search import *


class Example(QWidget):
    def __init__(self):
        super(Example, self).__init__()
        self.nameLineEdit = QLineEdit()
        self.labelList = []
        self.labelText = []
        t1 = "颜色特征选择：颜色直方图"
        t2 = "纹理特征选择：Gabor小波变换法"
        t3 = "形状特征选择：Canny算子边缘检测法"
        t4 = "resnet18深度学习算法"
        t5 = "DPSH深度哈希学习算法"
        self.btnFeatureSelect = QButtonGroup()
        # self.gridGroupBox6 = QGroupBox("检索进度")
        # self.pbar = QProgressBar(self)
        self.timer = QBasicTimer()
        self.step = 0
        self.gridGroupBox7 = QGroupBox("检测结果")
        self.gridGroupBox5 = QGroupBox("计算时间")
        self.label6 = QLabel("last time")
        self.label5 = QLabel("start time")
        self.label4 = QLabel()  # 显示当前时间
        self.label3 = QLabel("  持续时间：")
        self.label2 = QLabel("  开始时间：")
        self.label1 = QLabel("  当前时间：")
        self.cb3 = QComboBox()
        self.cb2 = QComboBox()
        self.cb1 = QComboBox()
        self.rbb2 = QRadioButton(t2, self)
        self.rbb3 = QRadioButton(t3, self)
        self.rbb1 = QRadioButton(t1, self)
        self.rbb4 = QRadioButton(t4, self)
        self.rbb5 = QRadioButton(t5, self)
        self.btnFeatureSelect.addButton(self.rbb1)
        self.btnFeatureSelect.addButton(self.rbb2)
        self.btnFeatureSelect.addButton(self.rbb3)
        self.btnFeatureSelect.addButton(self.rbb4)
        self.btnFeatureSelect.addButton(self.rbb5)
        # self.btnFeatureSelect.buttonClicked()
        self.gridGroupBoxCt = QGroupBox("单一特征选择")
        self.label = QLabel(self)
        self.gridGroupBoxP = QGroupBox()
        self.start_btn = QPushButton("开始检索", self)
        self.start_btn.clicked.connect(self.onStartRetrieval)
        self.gridGroupBox = QGroupBox("图像库选择")
        self.imagNames = ['default,png', 'default,png', 'default,png',
                          'default,png', 'default,png', 'default,png',
                          'default,png', 'default,png', 'default,png'
                          ]
        self.imgResult = []
        self.initUi()

    def initUi(self):
        # 页面整体布局
        self.resize(1000, 800)
        self.setWindowTitle("图像检索系统")
        self.createGridGroupBoxCP()
        self.createGridGroupBoxCT()
        self.createGridGroupBoxP()
        # self.createGridGroupBox4()
        self.createGridGroupBox5()
        # self.createGridGroupBox6()
        self.createGridGroupBox7()

        main_layout = QHBoxLayout()
        vbox_layout1 = QVBoxLayout()
        vbox_layout2 = QVBoxLayout()
        vbox_layout1.addWidget(self.gridGroupBox, 1)
        vbox_layout1.addWidget(self.gridGroupBoxP, 3)
        vbox_layout1.addWidget(self.gridGroupBoxCt, 3)
        # vbox_layout1.addWidget(self.gridGroupBox4, 2)
        vbox_layout1.addWidget(self.gridGroupBox5, 2)
        vbox_layout1.addWidget(self.start_btn)
        # vbox_layout2.addWidget(self.gridGroupBox6, 1)
        vbox_layout2.addWidget(self.gridGroupBox7, 11)
        main_layout.addLayout(vbox_layout1, 3)
        main_layout.addLayout(vbox_layout2, 7)
        self.setLayout(main_layout)

    def createGridGroupBoxCP(self):
        # 选择图像布局
        layout = QGridLayout()
        openBtn = QPushButton("选择图像", self)
        openBtn.clicked.connect(self.openImage)
        layout.setSpacing(1)
        layout.addWidget(self.nameLineEdit, 1, 0)
        layout.setSpacing(10)
        layout.addWidget(openBtn, 1, 1)
        self.gridGroupBox.setLayout(layout)

    def openImage(self):
        # 选择图像动作
        imgName, imgType = QFileDialog.getOpenFileName(self, "选择图像", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        self.nameLineEdit.setText(imgName)
        # return imgName

    def createGridGroupBoxP(self):
        # 图片显示布局
        layout = QGridLayout()
        self.label.setText("            显示图片")
        self.label.setFixedSize(300, 200)
        layout.addWidget(self.label)
        self.gridGroupBoxP.setLayout(layout)

    def createGridGroupBoxCT(self):
        # 特征选择布局
        layout = QGridLayout()
        # self.cb1.addItem('颜色直方图')
        # self.cb2.addItem('Gabor小波变换法')
        # self.cb3.addItem('Canny边缘检测算法')

        layout.addWidget(self.rbb1, 0, 1)
        layout.addWidget(self.rbb2, 1, 1)
        layout.addWidget(self.rbb3, 2, 1)
        layout.addWidget(self.rbb4, 3, 1)
        layout.addWidget(self.rbb5, 4, 1)
        self.gridGroupBoxCt.setLayout(layout)

    # def createGridGroupBox4(self):
    #     # 综合特征布局
    #
    #     # t1 = "综合特征选择"
    #     self.gridGroupBox4 = QGroupBox("综合特征权值设置")
    #     layout = QGridLayout()
    #     # self.rbb1 = QRadioButton(t1, self)
    #     layout.addWidget(self.rbb4)
    #     self.labely = QLabel("颜色")
    #     self.spinboxy = QSpinBox()
    #     self.spinboxy.setSingleStep(1)
    #     self.spinboxy.setRange(0, 10)
    #     self.labelw = QLabel("纹理")
    #     self.spinboxw = QSpinBox()
    #     self.labelx = QLabel("形状")
    #     self.spinboxx = QSpinBox()
    #     layout.addWidget(self.labely, 1, 1)
    #     layout.addWidget(self.spinboxy, 1, 2)
    #     layout.addWidget(self.labelw, 1, 3)
    #     layout.addWidget(self.spinboxw, 1, 4)
    #     layout.addWidget(self.labelx, 1, 5)
    #     layout.addWidget(self.spinboxx, 1, 6)
    #     self.gridGroupBox4.setLayout(layout)

    def createGridGroupBox5(self):
        # 计算时间布局
        layout = QGridLayout()
        self.init_timer()
        layout.addWidget(self.label1, 0, 1)
        layout.addWidget(self.label2, 1, 1)
        layout.addWidget(self.label3, 2, 1)
        layout.addWidget(self.label4, 0, 2)
        layout.addWidget(self.label5, 1, 2)
        layout.addWidget(self.label6, 2, 2)
        self.gridGroupBox5.setLayout(layout)

    def createGridGroupBox6(self):
        # 检索进度框
        layout = QGridLayout()
        self.pbar.setMaximum(100)
        self.pbar.setMinimum(0)
        self.pbar.setValue(50)  # 设置pbar的当前值
        layout.addWidget(self.pbar)
        self.gridGroupBox6.setLayout(layout)

    def createGridGroupBox7(self):
        # 9宫格布局
        layoutnine = QGridLayout()
        self.setNinePhoto()
        positions = [(i, j) for i in range(3) for j in range(3)]
        positions1 = [(0, j) for j in range(3)]
        positions2 = [(1, j) for j in range(3)]
        positions3 = [(2, j) for j in range(3)]
        positions4 = [(3, j) for j in range(3)]
        positions5 = [(4, j) for j in range(3)]
        positions6 = [(5, j) for j in range(3)]
        nums1 = [i for i in range(3)]
        nums2 = [i for i in range(3, 6)]
        nums3 = [i for i in range(6, 9)]
        for position, num in zip(positions1, nums1):
            layoutnine.addWidget(self.labelList[num], *position)
        for position, num in zip(positions2, nums1):
            layoutnine.addWidget(self.labelText[num], *position)
        for position, num in zip(positions3, nums2):
            layoutnine.addWidget(self.labelList[num], *position)
        for position, num in zip(positions4, nums2):
            layoutnine.addWidget(self.labelText[num], *position)
        for position, num in zip(positions5, nums3):
            layoutnine.addWidget(self.labelList[num], *position)
        for position, num in zip(positions6, nums3):
            layoutnine.addWidget(self.labelText[num], *position)

        self.gridGroupBox7.setLayout(layoutnine)

    def setNinePhoto(self):
        # positions = [(i, j) for i in range(3) for j in range(3)]
        nums = [i for i in range(9)]
        if len(self.labelList):
            # labellist内已经有数据，则替代该位置label
            for num, imgName, imgresult in zip(nums, self.imagNames, self.imgResult):
                jpg = QPixmap(imgName).scaled(200, 200)
                self.labelList[num].setPixmap(jpg)
                self.labelText[num].setText("置信度：" + str(imgresult))

        else:
            # 如果labellist为空，则显示初始数据
            for imgName in zip(self.imagNames):
                jpg = QPixmap('default.png').scaled(200, 200)
                label = QLabel()
                textLabel = QLabel()
                label.setPixmap(jpg)
                textLabel.setText("置信度：无")
                self.labelList.append(label)
                self.labelText.append(textLabel)

    def init_timer(self):
        # 提供了定时器信号和单出发定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showtime)
        self.timer.start(1000)  # 启动这个定时器

    def showtime(self):
        # 显示时间，设定显示格式
        # self.label4.setText(time.strftime('%H:%M:%S', time.localtime(time.time())))
        self.label4.setText(time.strftime(datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')))

    def timerEvent(self, a0: 'QTimerEvent'):
        if self.step >= 100:
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)

    def onStartRetrieval(self):
        print("开始检索")
        if self.btnFeatureSelect.checkedButton():
            print(self.btnFeatureSelect.checkedButton().text())
            # 判断所选特征
            if self.nameLineEdit.text() == '':
                # 未选择图像特征
                print("您还没有选择图像")
            else:
                # 检索开始时间
                startTime = datetime.datetime.now()
                t1 = time.time()
                self.label5.setText(startTime.strftime('%Y/%m/%d %H:%M:%S'))
                # self.label5.setText(time.strftime('%H:%M:%S', time.localtime(time.time())))
                # self.label5.setText(time.strftime('%H:%M:%S', self.startTime))
                # 判断所选图像特征，并调用相关方法

                if self.btnFeatureSelect.checkedButton().text() == 'resnet18深度学习算法':
                    # 所选为感知哈希算法
                    # print(searchColorFeature(self.nameLineEdit.text()))
                    # self.imagNames, self.imgResult = searchPHashFeature(self.nameLineEdit.text())
                    self.imagNames, self.imgResult = searchRestFeature(self.nameLineEdit.text())
                    self.setNinePhoto()
                elif self.btnFeatureSelect.checkedButton().text() == '形状特征选择：Canny算子边缘检测法':
                    # 所选为形状特征
                    # print(searchCannyFeature(self.nameLineEdit.text()))
                    self.imagNames, self.imgResult = searchCannyFeature(self.nameLineEdit.text())
                    self.setNinePhoto()
                elif self.btnFeatureSelect.checkedButton().text() == '纹理特征选择：Gabor小波变换法':
                    # 所选为纹理特征
                    print("纹理特征")
                    # print(searchTexturalFeature(self.nameLineEdit.text()))
                    self.imagNames, self.imgResult = searchGaborFeature(self.nameLineEdit.text())
                    self.setNinePhoto()
                elif self.btnFeatureSelect.checkedButton().text() == '颜色特征选择：颜色直方图':
                    # 所选为颜色特征
                    # print(searchTexturalFeature(self.nameLineEdit.text()))
                    self.imagNames, self.imgResult = searchHSVColorFeature(self.nameLineEdit.text())[0:9]
                    # self.imgResult = searchHSVColorFeature(self.nameLineEdit.text())[9:18]
                    self.setNinePhoto()
                elif self.btnFeatureSelect.checkedButton().text() == 'DPSH深度哈希学习算法':
                    # 所选为深度学习特征
                    # print(searchTexturalFeature(self.nameLineEdit.text()))
                    self.imagNames, self.imgResult = searchDeepFeature(48, self.nameLineEdit.text())
                    # self.imgResult = searchHSVColorFeature(self.nameLineEdit.text())[9:18]
                    self.setNinePhoto()
                # 检索结束时间
                # self.pbar.setValue(100)
                endTime = datetime.datetime.now()
                t2 = time.time()
                lasttime = (t2-t1)*1000
                # self.label6.setText(str((endTime - startTime).seconds) + "s")
                self.label6.setText(str(round(lasttime)) + "ms")
                print((endTime - startTime).seconds)

            # self.gridGroupBox7.update()
            # print(self.imagNames)
            # QApplication.processEvents()
            # time.sleep(1)
        else:
            print("您还没有选择特征")
        # self.setNinePhoto()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = Example()
    ex.show()
    sys.exit(app.exec_())
