import numpy as np
import pandas as pd
from PyQt5.QtGui import QPixmap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QFileDialog
from PyQt5 import QtCore, QtGui, QtWidgets
from may_hoc_giaodien import  Ui_MainWindow
import cv2
import numpy as np
import Preprocess
import math

class MainWindow:
    def __init__(self):
        self.thread = None
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.uic.btn_img.clicked.connect(self.BrowserImg)
        self.uic.btn_img_detec.clicked.connect(self.Detected_img)
        self.uic.btn_vid.clicked.connect(self.BrowserVid)
        self.uic.btn_vid_detec.clicked.connect(self.Detected_Vid)
        self.uic.btn_real.clicked.connect(self.Detected_Real)

        self.thread={};
    def show(self):
        self.main_win.show()

    def BrowserImg(self):
        link= QFileDialog.getOpenFileName(filter='*.jpg *.png')
        self.uic.img.setPixmap(QPixmap(link[0]))
        global linking
        linking= link[0]

    def Detected_img(self):

        ADAPTIVE_THRESH_BLOCK_SIZE = 19
        ADAPTIVE_THRESH_WEIGHT = 9

        n = 1

        Min_char = 0.01
        Max_char = 0.09

        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30

        img = cv2.imread(linking)
        img = cv2.resize(img, dsize=(1920, 1080))

        ###################### So sánh việc sử dụng độ tương phản#############
        # img2 = cv2.imread("1.jpg")
        # imgGrayscaleplate2, _ = Preprocess.preprocess(img)
        # imgThreshplate2 = cv2.adaptiveThreshold(imgGrayscaleplate2, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE ,ADAPTIVE_THRESH_WEIGHT )
        # cv2.imshow("imgThreshplate2",imgThreshplate2)
        ###############################################################

        ######## Tải lên mô hình KNN ######################
        npaClassifications = np.loadtxt("classificationS.txt",
                                        np.float32)  # classifications.txt có nhiệm vụ lưu các mã ASCII của các kí tự đó
        npaFlattenedImages = np.loadtxt("flattened_images.txt",
                                        np.float32)  # flattened_images.txt sẽ lưu giá trị các điểm ảnh có trong hình ảnh kí tự (hình 20x30 pixel có tổng cộng 600 điểm ảnh có giá trị 0 hoặc 255)
        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
        kNearest = cv2.ml.KNearest_create()  # khởi tạo KNN object
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
        #########################

        ################ Tiền xử lý ảnh #################
        imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
        canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tách biên bằng canny
        kernel = np.ones((3, 3), np.uint8)
        dilated_image = cv2.dilate(canny_image, kernel, iterations=1)  # tăng sharp cho egde (Phép nở)
        # cv2.imshow("dilated_image",dilated_image)

        ###########################################

        ###### vẽ contour và lọc biển số  #############
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất

        # cv2.drawContours(img, contours, -1, (255, 0, 255), 3) # Vẽ tất cả các ctour trong hình lớn

        screenCnt = []
        for c in contours:
            peri = cv2.arcLength(c, True)  # Tính chu vi
            approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
            [x, y, w, h] = cv2.boundingRect(approx.copy())
            ratio = w / h
            # cv2.putText(img, str(len(approx.copy())), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
            # cv2.putText(img, str(ratio), (x,y),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
            if (len(approx) == 4):
                screenCnt.append(approx)

                cv2.putText(img, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        if screenCnt is None:
            detected = 0
            print("No plate detected")
        else:
            detected = 1

        if detected == 1:

            for screenCnt in screenCnt:
                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

                ############## Tìm góc xoay ảnh #####################
                (x1, y1) = screenCnt[0, 0]
                (x2, y2) = screenCnt[1, 0]
                (x3, y3) = screenCnt[2, 0]
                (x4, y4) = screenCnt[3, 0]
                array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                (x1, y1) = array[0]
                (x2, y2) = array[1]
                doi = abs(y1 - y2)
                ke = abs(x1 - x2)
                angle = math.atan(doi / ke) * (180.0 / math.pi)

                ####################################

                ########## Cắt biển số ra khỏi ảnh và xoay ảnh ################

                mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
                # cv2.imshow("new_image",new_image)
                # Now crop
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))

                roi = img[topx:bottomx, topy:bottomy]
                imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
                ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                if x1 < x2:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                else:
                    rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
                roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                ####################################

                #################### Tiền xử lý ảnh đề phân đoạn kí tự ####################
                kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                cv2.imshow(str(n + 20), thre_mor)
                cv2.drawContours(roi, cont, -1, (100, 255, 255), 2)  # Vẽ contour các kí tự trong biển số

                ##################### Lọc vùng kí tự #################
                char_x_ind = {}
                char_x = []
                height, width, _ = roi.shape
                roiarea = height * width

                for ind, cnt in enumerate(cont):
                    (x, y, w, h) = cv2.boundingRect(cont[ind])
                    ratiochar = w / h
                    char_area = w * h
                    # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)
                    # cv2.putText(roi, str(ratiochar), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

                    if (Min_char * roiarea < char_area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                        if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                            x = x + 1
                        char_x.append(x)
                        char_x_ind[x] = ind

                        # cv2.putText(roi, str(char_area), (x, y+20),cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 2)

                ############ Cắt và nhận diện kí tự ##########################

                char_x = sorted(char_x)
                strFinalString = ""
                first_line = ""
                second_line = ""

                for i in char_x:
                    (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình

                    imgROIResized = cv2.resize(imgROI,
                                               (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
                    npaROIResized = imgROIResized.reshape(
                        (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều
                    # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                    npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                    _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                            k=5)  # call KNN function find_nearest; neigh_resp là hàng xóm
                    strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
                    cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0), 3)

                    if (y < height / 3):  # Biển số 1 hay 2 hàng
                        first_line = first_line + strCurrentChar
                    else:
                        second_line = second_line + strCurrentChar

                print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")
                self.uic.txt_img.setText("\n License Plate " + str(n) + " is: \n " + first_line + " - " + second_line + "\n"+self.uic.txt_img.toPlainText())
                roi = cv2.resize(roi, None, fx=0.75, fy=0.75)
                cv2.imshow(str(n), cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                # cv2.putText(img, first_line + "-" + second_line ,(topy ,topx),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)
                n = n + 1

        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        cv2.imshow('License plate', img)

        cv2.waitKey(0)

    def BrowserVid(self):
        link= QFileDialog.getOpenFileName(filter='*.mp4 *.avi')
        self.uic.img.setPixmap(QPixmap(link[0]))
        global linkVid
        linkVid= link[0]

    def Detected_Vid(self):
        ADAPTIVE_THRESH_BLOCK_SIZE = 19
        ADAPTIVE_THRESH_WEIGHT = 9

        Min_char_area = 0.015
        Max_char_area = 0.06

        Min_char = 0.01
        Max_char = 0.09

        Min_ratio_char = 0.25
        Max_ratio_char = 0.7

        max_size_plate = 18000
        min_size_plate = 5000

        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30

        tongframe = 0
        biensotimthay = 0

        # Load KNN model
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
        kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        # Đọc video
        cap = cv2.VideoCapture(linkVid)
        while (cap.isOpened()):

            # tiền xử lý ảnh
            ret, img = cap.read()
            if ret == True:
                tongframe = tongframe + 1
                # img = cv2.resize(img, None, fx=0.5, fy=0.5)
                imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
                canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tách biên bằng canny
                kernel = np.ones((3, 3), np.uint8)
                dilated_image = cv2.dilate(canny_image, kernel,
                                           iterations=1)  # tăng sharp cho egde (Phép nở). để biên canny chỗ nào bị đứt thì nó liền lại để vẽ contour

                # lọc vùng biển số
                contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
                screenCnt = []
                for c in contours:
                    peri = cv2.arcLength(c, True)  # Tính chu vi
                    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
                    [x, y, w, h] = cv2.boundingRect(approx.copy())
                    ratio = w / h
                    if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
                        screenCnt.append(approx)
                if screenCnt is None:
                    detected = 0
                    print("No plate detected")
                else:
                    detected = 1

                if detected == 1:
                    n = 1
                    for screenCnt in screenCnt:

                        ################## Tính góc xoay###############
                        (x1, y1) = screenCnt[0, 0]
                        (x2, y2) = screenCnt[1, 0]
                        (x3, y3) = screenCnt[2, 0]
                        (x4, y4) = screenCnt[3, 0]
                        array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                        (x1, y1) = array[0]
                        (x2, y2) = array[1]

                        doi = abs(y1 - y2)
                        ke = abs(x1 - x2)
                        angle = math.atan(doi / ke) * (180.0 / math.pi)
                        #################################################

                        # Masking the part other than the number plate
                        mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

                        # Now crop
                        (x, y) = np.where(mask == 255)
                        (topx, topy) = (np.min(x), np.min(y))
                        (bottomx, bottomy) = (np.max(x), np.max(y))

                        roi = img[topx:bottomx + 1, topy:bottomy + 1]
                        imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

                        ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                        if x1 < x2:
                            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                        else:
                            rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                        roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                        imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

                        roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                        imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                        # Tiền xử lý biển số

                        kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                        thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                        cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # Phân đoạn kí tự
                        char_x_ind = {}
                        char_x = []
                        height, width, _ = roi.shape
                        roiarea = height * width
                        # print ("roiarea",roiarea)
                        for ind, cnt in enumerate(cont):
                            area = cv2.contourArea(cnt)
                            (x, y, w, h) = cv2.boundingRect(cont[ind])
                            ratiochar = w / h
                            if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                                if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                                    x = x + 1
                                char_x.append(x)
                                char_x_ind[x] = ind

                        # Nhận diện kí tự và in ra số xe
                        if len(char_x) in range(7, 10):
                            cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                            char_x = sorted(char_x)
                            strFinalString = ""
                            first_line = ""
                            second_line = ""

                            for i in char_x:
                                (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                                cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình

                                imgROIResized = cv2.resize(imgROI,
                                                           (RESIZED_IMAGE_WIDTH,
                                                            RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
                                npaROIResized = imgROIResized.reshape(
                                    (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều
                                # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                                npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                                _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                                        k=5)  # call KNN function find_nearest; neigh_resp là hàng xóm
                                strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
                                cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                                if (y < height / 3):  # Biển số 1 hay 2 hàng
                                    first_line = first_line + strCurrentChar
                                else:
                                    second_line = second_line + strCurrentChar

                            strFinalString = first_line + second_line
                            print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")

                            self.uic.txt_img.setText("\n License Plate " + str(n) + " is: \n " + first_line + " - " + second_line + "\n"+self.uic.txt_img.toPlainText())
                            cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                            n = n + 1
                            biensotimthay = biensotimthay + 1

                            cv2.imshow("a", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                imgcopy = cv2.resize(img, (1280, 720), fx=0.5, fy=0.5)
                cv2.imshow('License plate', imgcopy)
                print("biensotimthay", biensotimthay)
                print("tongframe", tongframe)
                print("ti le tim thay bien so:", 100 * biensotimthay / (368), "%")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def Detected_Real(self):
        ADAPTIVE_THRESH_BLOCK_SIZE = 19
        ADAPTIVE_THRESH_WEIGHT = 9

        Min_char_area = 0.015
        Max_char_area = 0.06

        Min_char = 0.01
        Max_char = 0.09

        Min_ratio_char = 0.25
        Max_ratio_char = 0.7

        max_size_plate = 18000
        min_size_plate = 5000

        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 30

        tongframe = 0
        biensotimthay = 0

        # Load KNN model
        npaClassifications = np.loadtxt("classifications.txt", np.float32)
        npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
        npaClassifications = npaClassifications.reshape(
            (npaClassifications.size, 1))  # reshape numpy array to 1d, necessary to pass to call to train
        kNearest = cv2.ml.KNearest_create()  # instantiate KNN object
        kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)

        # Đọc video
        cap = cv2.VideoCapture(0)
        while (cap.isOpened()):

            # tiền xử lý ảnh
            ret, img = cap.read()
            tongframe = tongframe + 1
            # img = cv2.resize(img, None, fx=0.5, fy=0.5)
            imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(img)
            canny_image = cv2.Canny(imgThreshplate, 250, 255)  # Tách biên bằng canny
            kernel = np.ones((3, 3), np.uint8)
            dilated_image = cv2.dilate(canny_image, kernel,
                                       iterations=1)  # tăng sharp cho egde (Phép nở). để biên canny chỗ nào bị đứt thì nó liền lại để vẽ contour

            # lọc vùng biển số
            contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # Lấy 10 contours có diện tích lớn nhất
            screenCnt = []
            for c in contours:
                peri = cv2.arcLength(c, True)  # Tính chu vi
                approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
                [x, y, w, h] = cv2.boundingRect(approx.copy())
                ratio = w / h
                if (len(approx) == 4) and (0.8 <= ratio <= 1.5 or 4.5 <= ratio <= 6.5):
                    screenCnt.append(approx)
            if screenCnt is None:
                detected = 0
                print("No plate detected")
            else:
                detected = 1

            if detected == 1:
                n = 1
                for screenCnt in screenCnt:

                    ################## Tính góc xoay###############
                    (x1, y1) = screenCnt[0, 0]
                    (x2, y2) = screenCnt[1, 0]
                    (x3, y3) = screenCnt[2, 0]
                    (x4, y4) = screenCnt[3, 0]
                    array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    sorted_array = array.sort(reverse=True, key=lambda x: x[1])
                    (x1, y1) = array[0]
                    (x2, y2) = array[1]

                    doi = abs(y1 - y2)
                    ke = abs(x1 - x2)
                    angle = math.atan(doi / ke) * (180.0 / math.pi)
                    #################################################

                    # Masking the part other than the number plate
                    mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
                    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )

                    # Now crop
                    (x, y) = np.where(mask == 255)
                    (topx, topy) = (np.min(x), np.min(y))
                    (bottomx, bottomy) = (np.max(x), np.max(y))

                    roi = img[topx:bottomx + 1, topy:bottomy + 1]
                    imgThresh = imgThreshplate[topx:bottomx + 1, topy:bottomy + 1]

                    ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

                    if x1 < x2:
                        rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
                    else:
                        rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

                    roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
                    imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))

                    roi = cv2.resize(roi, (0, 0), fx=3, fy=3)
                    imgThresh = cv2.resize(imgThresh, (0, 0), fx=3, fy=3)

                    # Tiền xử lý biển số

                    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thre_mor = cv2.morphologyEx(imgThresh, cv2.MORPH_DILATE, kerel3)
                    cont, hier = cv2.findContours(thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Phân đoạn kí tự
                    char_x_ind = {}
                    char_x = []
                    height, width, _ = roi.shape
                    roiarea = height * width
                    # print ("roiarea",roiarea)
                    for ind, cnt in enumerate(cont):
                        area = cv2.contourArea(cnt)
                        (x, y, w, h) = cv2.boundingRect(cont[ind])
                        ratiochar = w / h
                        if (Min_char * roiarea < area < Max_char * roiarea) and (0.25 < ratiochar < 0.7):
                            if x in char_x:  # Sử dụng để dù cho trùng x vẫn vẽ được
                                x = x + 1
                            char_x.append(x)
                            char_x_ind[x] = ind

                    # Nhận diện kí tự và in ra số xe
                    if len(char_x) in range(7, 10):
                        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

                        char_x = sorted(char_x)
                        strFinalString = ""
                        first_line = ""
                        second_line = ""

                        for i in char_x:
                            (x, y, w, h) = cv2.boundingRect(cont[char_x_ind[i]])
                            cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                            imgROI = thre_mor[y:y + h, x:x + w]  # cắt kí tự ra khỏi hình

                            imgROIResized = cv2.resize(imgROI,
                                                       (RESIZED_IMAGE_WIDTH,
                                                        RESIZED_IMAGE_HEIGHT))  # resize lại hình ảnh
                            npaROIResized = imgROIResized.reshape(
                                (1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))  # đưa hình ảnh về mảng 1 chiều
                            # cHUYỂN ảnh thành ma trận có 1 hàng và số cột là tổng số điểm ảnh trong đó
                            npaROIResized = np.float32(npaROIResized)  # chuyển mảng về dạng float
                            _, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                                    k=5)  # call KNN function find_nearest; neigh_resp là hàng xóm
                            strCurrentChar = str(chr(int(npaResults[0][0])))  # Lấy mã ASCII của kí tự đang xét
                            cv2.putText(roi, strCurrentChar, (x, y + 50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

                            if (y < height / 3):  # Biển số 1 hay 2 hàng
                                first_line = first_line + strCurrentChar
                            else:
                                second_line = second_line + strCurrentChar

                        strFinalString = first_line + second_line
                        print("\n License Plate " + str(n) + " is: " + first_line + " - " + second_line + "\n")

                        self.uic.txt_img.setText("\n License Plate " + str(n) + " is: \n " + first_line + " - " + second_line + "\n"+self.uic.txt_img.toPlainText())
                        cv2.putText(img, strFinalString, (topy, topx), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
                        n = n + 1
                        biensotimthay = biensotimthay + 1

                        cv2.imshow("a", cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

            imgcopy = cv2.resize(img, (1280, 720), fx=0.5, fy=0.5)
            cv2.imshow('License plate', imgcopy)
            print("biensotimthay", biensotimthay)
            print("tongframe", tongframe)
            print("ti le tim thay bien so:", 100 * biensotimthay / (368), "%")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())