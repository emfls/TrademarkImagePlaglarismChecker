# App.py

# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf


class ImageCheck:
    image_path = None
    model_full_path = 'Data/retrained_graph.pb'
    labels_full_path = 'Data/retrained_labels.txt'
    human_string = None
    score = None
    result = []
    answer = None


    # tensorflow Warning 문구 제거
    # for Mac OS
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    ##

    def __init__(self, image_path):
        print('init')
        self.image_path = image_path

    def create_graph(self):
        print('create graph')
        # 저장된(saved) retrained_graph.pb로부터 graph를 생성한다.
        with tf.gfile.FastGFile(self.model_full_path, 'rb') as f:  # 바이너리 형식으로 읽어들임
            graph_def = tf.GraphDef()  # 그래프를 로딩하고 조작하기 위해 객체 생성
            graph_def.ParseFromString(f.read())  # 바이너리 파일을 읽기위해 사용
            _ = tf.import_graph_def(graph_def, name='')

    def run_inference_on_image(self):
        print('run')
        self.answer = None
        self.result = []

        test_image = self.image_path

        ## no image ##
        if not tf.gfile.Exists(test_image):
            print('no image')
            tf.logging.fatal('File does not exist %s', test_image)
            ## 상태바에도 출력하도록 구현해야함
            return self.answer
        ## ##
        print('Image Set. "' + self.image_path + '"')



        # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
        self.create_graph()

        with tf.Session() as sess:
            print('session start')
            image_data = tf.gfile.FastGFile(test_image, 'rb').read()
            predictions = sess.run('final_result:0', {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
            f = open(self.labels_full_path, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\\r\\n", "") for w in lines]

            order = 1
            for node_id in top_k:
                human_string = labels[node_id]
                human_string = human_string.split("'")
                if order is 1:
                    firstScore = 1
                score = predictions[node_id]
                result = [order, human_string[1], score * 100]  # rank, label, score
                self.result.append(result) # result 리스트에 결과 문장 저장.
                order += 1

            self.answer = labels[top_k[0]]

            print(self.answer)
            self.answer = self.answer[2:len(self.answer)-1]

            print(result)

            print(self.result)
            print('Test End.')

            print("Similarity Rank...")
            for result in self.result[0:len(self.result)]:
                # rank #
                if result[0] is 1:
                    rank = '1st'
                elif result[0] is 2:
                    rank = '2nd'
                elif result[0] is 3:
                    rank = '3rd'
                else:
                    rank = str(result[0]) + 'st'

                # label #
                label = result[1]

                # similarity degree(percentage) #
                percentage = '%.2f' % result[2] + '%%'

                print(' %s: %s -> %s' % (rank, label, percentage))
            print("Most Similar: %s(%.2f%%)" % (self.result[0][1], self.result[0][2]))

            print('ImageChecking End.')
            return self.result






###############################################
## 유사도 검사 부분 코드 끝
###############################################


###############################################
## GUI Part
###############################################



# Divided from '../second/TrademarkCheckApp.py', Since 17.11.07
#
# Created by: Jho
#
#


import os, imghdr, time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PIL import Image
from PyQt5 import QtCore, QtGui,  QtWidgets



### MyWidget ###
class MyWidget(QWidget):
    image_path = None
    centralWidget = None
    imgView = None
    verticalLayout_widget = None
    btns_container = None
    help_btn = None
    openImage_btn = None
    startChecking_btn = None
    initialize_btn = None
    log_textbox = None


    def __init__(self):
        super().__init__()
        self.setupUI()


    ## setupUI ##
    def setupUI(self):
        ## Window ##
        self.setWindowTitle("Trademark Image Plagiarism Checker")
        self.setFixedSize(470, 547)
        ## End of Window ##

        ## centralWidget ##
        self.centralWidget = QtWidgets.QWidget(self)
        self.centralWidget.setObjectName("centralWidget")
        ## End of centralWidget ##

        ## imgView ##
        self.imgView = QtWidgets.QLabel(self.centralWidget)
        self.imgView.setGeometry(QtCore.QRect(10, 10, 300, 300))
        self.imgView.setObjectName("imgView")
        self.imgView.setPixmap(QtGui.QPixmap(os.getcwd().replace('\\', '/') + "Data/default_image.jpg"))
        ## End of imgView

        ## verticalLayout_widget ##
        self.verticalLayout_widget = QtWidgets.QWidget(self.centralWidget)
        self.verticalLayout_widget.setGeometry(QtCore.QRect(320, 10, 140, 300))
        self.verticalLayout_widget.setObjectName("verticalLayout_widget")
        ## End of verticalLayout_widget ##

        ## layout for buttons ##
        self.btns_container = QtWidgets.QVBoxLayout(self.verticalLayout_widget)
        self.btns_container.setContentsMargins(11, 11, 11, 11)
        self.btns_container.setSpacing(8)
        self.btns_container.setObjectName("horizontalLayout")
        ## End of layout for buttons ##

        ### Buttons ###
        ## help_btn ##
        self.help_btn = QPushButton("Help")
        self.help_btn.setObjectName("help_btn")
         ## 도움말 출력 부분.. 미구현 ##
        self.btns_container.addWidget(self.help_btn)
        ## End of help_btn ##

        ## openImage_btn ##
        self.openImage_btn = QPushButton("Open Image File")
        self.openImage_btn.setObjectName("startChecking_btn")

        self.btns_container.addWidget(self.openImage_btn)
        ## End of openImage_btn ##

        ## startChecking_btn ##
        self.startChecking_btn = QtWidgets.QPushButton("Start Checking")
        self.startChecking_btn.setObjectName("startChecking_btn")
        self.btns_container.addWidget(self.startChecking_btn)
        ## End of startChecking_btn ##

        ## save_btn ##
        self.save_btn = QtWidgets.QPushButton("Save Log into txt")
        self.save_btn.setObjectName("save_btn")
        self.btns_container.addWidget(self.save_btn)
        ## End of save_btn ##

        ## initialize_btn ##
        self.initialize_btn = QtWidgets.QPushButton("Initialize")

        # self.initialize_btn.setIcon(QtGui.QIcon('../images/icon/Cancel.png'))
        # self.initialize_btn.setIconSize(QtCore.QSize(300, 30))
        self.initialize_btn.setObjectName("initialize_btn")
        self.btns_container.addWidget(self.initialize_btn)
        ## End of initialize_btn ##

        ### End of Buttons ###

        ## log_textbox ##
        self.log_textbox = QPlainTextEdit(self.centralWidget)
        self.log_textbox.setGeometry(QtCore.QRect(20, 330, 430, 120))
        self.log_textbox.setReadOnly(True)

        ## End of log_textbox ##
    ## End of setupUI ##







### End of MyWidget ###

### MyMainWindow ###
class MyMainWindow(QMainWindow):
    status_message = ""
    wg = None
    myStatusBar = None
    file_name = None
    resultFromTest = None
    default_image = "Data/default_image.jpg"
    icon_image = "Data/icon.png"



    def __init__(self):
        super().__init__()
        self.wg = MyWidget()
        self.setupUI()
        self.wg.help_btn.clicked.connect(self.help_btn_clicked)
        self.wg.openImage_btn.clicked.connect(self.OpenFileDialog_btn_Clicked)
        self.wg.startChecking_btn.clicked.connect(self.startTest)
        self.wg.save_btn.clicked.connect(self.save_log)
        self.wg.initialize_btn.clicked.connect(self.initialize)
        self.initialize()
        self.set_status("Ready.")
        self.stamp_print("Ready.")


    ## setupUI ##
    def setupUI(self):
        ## Icon ##
        app_icon = QtGui.QIcon()
        app_icon.addFile(self.icon_image, QtCore.QSize(16, 16))
        app_icon.addFile(self.icon_image, QtCore.QSize(24, 24))
        app_icon.addFile(self.icon_image, QtCore.QSize(32, 32))
        app_icon.addFile(self.icon_image, QtCore.QSize(48, 48))
        app_icon.addFile(self.icon_image, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        ## ##

        ## Window Size and Location ##
        self.setGeometry(QtCore.QRect(500, 300, 470, 500))
        self.setFixedSize(470, 500)
        self.setWindowTitle("Trademark Image Plagiarism Checker")
        ## ##

        ## centralWidget ##
        self.setCentralWidget(self.wg)
        ## End of centralWidget ##

        ## statusbar ##
        self.myStatusBar = QStatusBar()
        self.setStatusBar(self.myStatusBar)
        self.set_status("Ready.")
        ## End of statusbar ##

    ## End of setupUI ##

    ## Initializing ##
    def initialize(self):
        self.wg.imgView.setPixmap(QPixmap(self.default_image))
        self.wg.imgView.setScaledContents(True)
        self.wg.log_textbox.clear()
        self.wg.image_path = None
        self.resultFromTest = None
        self.set_status("Initialized")
        self.stamp_print("Initialized.")

    ## End of Initializing ##

    ## Set a Message in the status bar ##
    def set_status(self, message):
        self.status_message = self.timeStamp() + message
        self.myStatusBar.showMessage(self.status_message)
    ## ##


    ## getFileName ##
    def OpenFileDialog_btn_Clicked(self):
        self.stamp_print("Open Button Clicked.")
        file_name = QFileDialog.getOpenFileName(self.wg, 'Open Image File', 'TestImage', "Image files (*.jpg *.png)")[0]
        if file_name == '':

            self.stamp_print("No File Loaded..")
            self.set_status("No File Loaded")
            self.wg.setupUI()
        else:
            self.wg.image_path = file_name
            self.stamp_print('File Path: ' + file_name)
            image_format = self.check_imageFormat(self.wg.image_path)
            if image_format is None:
                self.stamp_print("The file is not an image. \"%s\"" % self.wg.image_path)
                self.stamp_appendTextBox("The file is not an image. \"%s\"" % self.wg.image_path)
                self.set_status("File ERROR")
            else:
                self.stamp_appendTextBox("File Loaded. \"%s\"" % (self.wg.image_path))
                self.stamp_print('Format: ' + image_format)
                self.wg.image_path = self.convert_imageFile(self.wg.image_path)
                self.wg.imgView.setPixmap(QPixmap(file_name))

                self.wg.imgView.setScaledContents(True)
                self.set_status("File Loaded")
                self.stamp_print("File Loaded")



        return
    ## End of getFileName ##

    ## startTest ##
    def startTest(self):
        self.stamp_print("Start Test Button Clicked.")
        # 로딩된 파일이 없을 경우...
        if (self.wg.image_path is None):
            self.stamp_print("No Image is loaded.")
            self.set_status("Need 'Image Loading'")
        # 로딩된 파일이 있을 경우...
        else:
            Tester = None
            Tester = ImageCheck(self.wg.image_path)
            self.stamp_appendTextBox('Test Start.')
            self.set_status("Testing...")
            #test.create_graph()
            self.resultFromTest = None
            self.resultFromTest = []
            self.resultFromTest = Tester.run_inference_on_image()
            self.stamp_print("Test Session Ended.")
            self.stamp_print('Start printing result')
            # 결과 출력 #
            for result in self.resultFromTest[0:len(self.resultFromTest)]:
                # rank #
                if result[0] is 1:
                    rank = '1st'
                elif result[0] is 2:
                    rank = '2nd'
                elif result[0] is 3:
                    rank = '3rd'
                else:
                    rank = str(result[0]) + 'st'

                # label #
                label = result[1]

                # similarity degree(percentage) #
                percentage = '%.2f' % result[2] + '%'

                result_line = ' %s: %s -> %s' % (rank, label, percentage)

                self.wg.log_textbox.appendPlainText(result_line)
            self.stamp_appendTextBox("Most Similar: %s(%.2f%%)" % ( self.resultFromTest[0][1], self.resultFromTest[0][2]))
            self.stamp_print('Printing result ended.')
            # #

            self.autoSave_file(self.wg.log_textbox.toPlainText())

            self.set_status("Test End")

            self.stamp_appendTextBox('Test End.')
            self.wg.log_textbox.appendPlainText('')



    ## End of startTest ##


    ## Check Format of image ##
    def check_imageFormat(self, imagePath):

        return imghdr.what(imagePath)
    ## ##

    ## Time Stamp ##
    def timeStamp(self):
        stamp = time.strftime('[%y/%m/%d %H:%M:%S] ')
        return stamp

    ## ##

    def convert_imageFile(self, image_path):

        png = Image.open(image_path).convert('RGBA')
        png.load()

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3])  # 3 is the alpha channel

        image_dir = os.path.abspath('./TestImage').replace("\\", "/")

        self.create_dir(image_dir)

        test_image = time.strftime(image_dir + '/TestImage_%y%m%d%H%M%S.jpg')

        background.save(test_image, 'JPEG', quality=100)

        self.stamp_print('Image Converted and Saved. "%s" -> "%s"' %(image_path, test_image))
        self.stamp_appendTextBox('Test Image Saved in "%s"' %(test_image))
        return test_image

    def stamp_print(self, string):
        print(self.timeStamp() + string)
        return

    def stamp_appendTextBox(self, string):
        self.wg.log_textbox.appendPlainText(self.timeStamp() + string)
        return

    ## Save Log ##
    def save_log(self):
        self.stamp_print("Save Log Button Clicked.")
        log = self.wg.log_textbox.toPlainText()
        if log is not "":
            self.save_file(log)
        else:
            self.stamp_print("Log is empty.")
            self.set_status("No log data to save")
    ## End of Save Log ##

    ## Saving TXT File ##

    def save_file(self, txt):
        log_dir = os.path.abspath('./Log').replace("\\", "/")

        self.stamp_print('log directory: "' + log_dir + "'")

        # './Logs' 디렉토리 미존재시, 생성
        self.create_dir(log_dir)

        log_path = time.strftime(log_dir + '/TestLog_%y%m%d%H%M%S.txt')

        log_path, _ = QFileDialog.getSaveFileName(self, 'Save Log into TXT', log_path, "TXT Files (*.txt);;All Files (*)")
        if log_path is not '':
            self.stamp_print('Saving path: "' + log_path + '"')
            log_file = open(log_path, 'w+t', encoding="utf-8")
            log_file.write(txt)
            log_file.close()
            self.stamp_print('Log Saved: "' + log_path + '"')
            self.stamp_appendTextBox('Log Saved: "' + log_path + '"')
            self.set_status('Log Saved')
        else:
            self.stamp_print("No Path Selected...")
            self.set_status("Saving Canceled")

    ## End of Saving TXT File ##

    ## Auto Saving Log ##
    def autoSave_file(self, txt):

        log_dir = os.path.abspath('./Log').replace("\\", "/")
        self.stamp_print('log directory: "' + log_dir + "'")

        # './Logs' 디렉토리 미존재시, 생성
        self.create_dir(log_dir)

        autoSave_dir = (log_dir + '/AutoSaved')

        # './Logs/autoSaved' 디렉토리 미존재시, 생성
        self.create_dir(autoSave_dir)

        autoSave_path = time.strftime(autoSave_dir + '/AutoSaved_%y%m%d%H%M%S.txt')

        # Log Saving (auto)
        self.stamp_print('Log Auto Saving to.. "' + autoSave_path + '"')
        log_file = open(autoSave_path, 'w+t', encoding="utf-8")
        log_file.write(txt)
        log_file.close()
        self.stamp_print('Log Auto Saved: "' + autoSave_path + '"')
    ## End of Auto Saving Log ##


    # 미존재 디렉토리 생성 #
    def create_dir(self, _dir):
        _dir = _dir.replace('\\', '/')
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
            self.stamp_print('Directory was created. "' + _dir + '"')
        else:
            self.stamp_print('Directory is already exist. "' + _dir + '"')
    # #

    # Help Button Clicked #
    def help_btn_clicked(self):
        self.stamp_print('Help Button Clicked.')
        manual = os.path.abspath('./Data/user_manual.txt').replace("\\", "/")
        notepad = "c:\windows\system32\\notepad.exe "

        self.stamp_print('Manual Opened with Notepad. ' + '"' + manual + '"')

        self.set_status("Manual Opened.")
        os.system(notepad + manual)
        self.set_status("Manual Closed.")

        self.stamp_print('Manual Closed. ')


        return

    def setFocusOnBottom(self):
        c = self.wg.log_textbox.textCursor()
        c.movePosition(QTextCursor.atBlockEnd())
        self.wg.log_textbox.setTextCursor(c)
        return

### End of MyMainWindow ###


###############################################
## GUI 부분 끝
###############################################


import sys


### main ###
def main():
    app = QApplication(sys.argv)
    window = MyMainWindow()
    window.setupUI()
    window.show()
    sys.exit(app.exec_())
### End of Main ###


main()
