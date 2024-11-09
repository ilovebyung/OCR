import cv2
import pytesseract
import matplotlib.pyplot as plt
import os
import shutil

folder_names = ['5', '6', '7', '8', '9']

def create_folders(folder_names):
    for folder_name in folder_names:
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)


def read_conf(image_path):

    image = cv2.imread(image_path,0)
    inverted = cv2.bitwise_not(image)
    blur = cv2.GaussianBlur(inverted, (5, 5), 0) 

    # Thresholding to binarize the image
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    # Filter contours based on size and shape
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 200:  # Adjust the threshold as needed
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Crop the license plate region
            plate_image = inverted[y:y+h, x:x+w]

            # plt.imshow(plate_image)

            # Perform OCR on the cropped image
            string = pytesseract.image_to_string(plate_image, config='--psm 6')
            # print(string)
            if string.startswith("defect"):
                # Extract Confidence
                return string[9:10] 


def copy_jpg_files():
    for file in os.listdir():
        if file.endswith(".jpg"):
            conf = read_conf(file)
            if conf in folder_names:
                dst_file = os.path.join(conf, file)
                shutil.copy2(file, dst_file)
                print(f"{file} is copied to folder {conf} ")




if __name__ == "__main__":

    create_folders(folder_names)

    # conf = read_conf("6.jpg")

    copy_jpg_files()
