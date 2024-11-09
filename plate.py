import cv2
import pytesseract
import matplotlib.pyplot as plt

image_path = "6.jpg"

def recognize_plate(image_path):

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

            plt.imshow(plate_image)

            # Perform OCR on the cropped image
            text = pytesseract.image_to_string(plate_image, config='--psm 7 --oem 3')
            print(text)
            if text.startswith("defect"):
                return text

# Replace 'your_image.jpg' with your actual image path
text = recognize_plate("6.jpg")

