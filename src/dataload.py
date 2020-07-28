import cv2
import numpy as np
import sys
import os
import glob
from PIL import Image

import matplotlib.pyplot as plt


def load_data(img_path, cnt):
    # 가위 : 0, 바위 : 1, 보 : 2
    #number_of_data = 5778  # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size = 28
    color = 3
    # 이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs = np.zeros(cnt * img_size * img_size * color, dtype=np.int32).reshape(cnt, img_size, img_size, color)
    labels = np.zeros(cnt, dtype=np.int32)

    idx = 0
    for file in glob.iglob(img_path + '/scissor/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 0  # 가위 : 0
        idx = idx + 1

    for file in glob.iglob(img_path + '/rock/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 1  # 바위 : 1
        idx = idx + 1

    for file in glob.iglob(img_path + '/paper/*.jpg'):
        img = np.array(Image.open(file), dtype=np.int32)
        imgs[idx, :, :, :] = img  # 데이터 영역에 이미지 행렬을 복사
        labels[idx] = 2  # 보 : 2
        idx = idx + 1

    print("학습데이터(x_train)의 이미지 개수는", idx, "입니다.")
    return imgs, labels


