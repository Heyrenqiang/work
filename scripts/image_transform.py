import cv2

img = cv2.imread("1.jpg")
img_new = cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
cv2.imwrite("new.jpg",img_new)