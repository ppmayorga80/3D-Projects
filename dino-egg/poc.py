import cv2

if __name__ == '__main__':
    image = cv2.imread('path5.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)

    print(bw_image.shape)
