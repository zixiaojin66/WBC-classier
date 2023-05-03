import cv2
import numpy as np
from Functions import feature_extractor
from Functions import segmentation
import pickle
import streamlit as st
import pandas as pd
import time
import io
from PIL import Image
#-----------功能实现----------------
def seg(pic):
    # img = cv2.imread(path)  # loading image
    img = pic
    Nucleus_img, img_convex, img_ROC = segmentation(img)

    img1 = img.copy()
    for i in range(len(Nucleus_img)):
        for j in range(len(Nucleus_img[0])):
            if Nucleus_img[i][j] == 255:
                img1[i][j][0] = 255
                img1[i][j][1] = 0
                img1[i][j][2] = 0

    img2 = img.copy()
    for i in range(len(Nucleus_img)):
        for j in range(len(Nucleus_img[0])):
            if img_convex[i][j] == 255:
                img2[i][j][0] = 0
                img2[i][j][1] = 255
                img2[i][j][2] = 0

    img3 = img.copy()
    for i in range(len(Nucleus_img)):
        for j in range(len(Nucleus_img[0])):
            if img_ROC[i][j] == 255:
                img3[i][j][0] = 0
                img3[i][j][1] = 0
                img3[i][j][2] = 255

    return img,img1,img2,img3

def cal_wait():
    '分类中，请耐心等待...'

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'已完成 {i + 1} %')
        bar.progress(i + 1)
        time.sleep(0.01)
    '执行成功!'

def predict(img):
    label = ['淋巴细胞','单核细胞','中心粒细胞','嗜酸性粒细胞','嗜碱性粒细胞']
    # img = cv2.imread(path)  # loading image
    # Extracting shape and color features from image
    ncl_detect, error, ftrs = feature_extractor(img=img, min_area=100)
    # saving image feature vector to memory and appending to x and y lists
    mn= np.load(r'mn.npy')
    mx= np.load(r'mx.npy')
    ftrs = (ftrs - mn)/(mx - mn)
    with open('train_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    prd = clf.predict([ftrs])

    return label[prd[0]-1]
#-------------界面实现----------------
st.title("白细胞检测与分类系统🎈")
pic = st.sidebar.file_uploader("请选择图片！")
clicked = st.sidebar.button('开始检测！')

if pic !=None:
    st.sidebar.header('请检查图片是否正确！')
    st.sidebar.image(pic)
    pic = Image.open(io.BytesIO(pic.read()))
pic = np.array(pic)
if clicked:
    with st.container():
        #显示分割后的图片
        col1, col2, col3, col4 = st.columns(4)
        cal_wait()
        img,img1,img2,img3 = seg(pic)
        col1.image(img,caption='原图片')
        col2.image(img1,caption='分割的细胞核')
        col3.image(img2,caption='分割的细胞核的凸包')
        col4.image(img3,caption='分割的细胞质的代表')

    #显示预测结果
    result = predict(pic)
    '该白细胞的种类为：'
    st.header(result)

# 隐藏made with streamlit
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
