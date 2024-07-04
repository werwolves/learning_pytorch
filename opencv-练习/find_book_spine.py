import os
import cv2
img_dir = r'D:\projects\RFID\datasets\book_loc\test_dataset_2024_6_26\Images'
for i in os.listdir(img_dir):
    x = os.path.join(img_dir, i)
    img = cv2.imread(x)

    img_h,img_w = img.shape[:2]
    img[:int(0.5*img_h), :, :] = 0


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  

    #--- performing Otsu threshold ---
    ret,thresh1 = cv2.threshold(gray, 0, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    # cv2.imshow('thresh1', thresh1)


    #--- choosing the right kernel
    #--- kernel size of 3 rows (to join dots above letters 'i' and 'j')
    #--- and 10 columns to join neighboring letters in words and neighboring words
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
    # cv2.imshow('dilation', dilation)

    #---Finding contours ---
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    im2 = img.copy()
    for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    window_name = 'Image Window'  
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 使用NORMAL标志允许调整窗口大小  
    cv2.resizeWindow(window_name, 1000, 600)  # 设置窗口的初始宽度和高度  
    # cv2.imshow(window_name,img)
    cv2.imshow(window_name,thresh1)
    # cv2.imshow(window_name, dilation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()