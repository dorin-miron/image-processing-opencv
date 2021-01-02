import cv2 
import pytesseract
from nltk.corpus import words
from pyspark import SparkContext, SparkConf
import random
import numpy as np

images=['1', '2', '3', '4']
sc = SparkContext("local", "PySpark Word Count Exmaple")


def check_for_word(text):
    s = text.split(' ')
    for word in s:
        if word not in words.words():
            s.remove(word)
    return ' '.join(s)


def word_count(text):
    words = sc.parallelize(text.split(" "))
    wordCounts = words.map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b).sortBy(lambda a: a[1])
    return [''.join(str(i)) for i in wordCounts.collect()]


def extract_text_and_save(img, img_nr, flow_type):
    custom_config = r'--oem 3 --psm 6'
    textFromImage = pytesseract.image_to_string(img, config=custom_config)
    txt = word_count(check_for_word(textFromImage))
    txt.reverse()
    file = open(f'data/{img_nr}/{flow_type}.txt', 'w')
    for line in txt:
        file.writelines(line)
        file.write("\n")
    file.close()


def read_image(img_path):
    img = cv2.imread(img_path, 1)
    return img


def add_noise(img):
    row, col, _ = img.shape
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 255
    number_of_pixels = random.randint(300, 10000)
    for i in range(number_of_pixels):
        y_coord = random.randint(0, row - 1)
        x_coord = random.randint(0, col - 1)
        img[y_coord][x_coord] = 0
    return img

def normal(img):
    return img


def affine_transformation(img):
    rows, cols, ch = img.shape

    pts1 = np.float32([[50, 50],
                       [200, 50],
                       [50, 200]])

    pts2 = np.float32([[30, 70],
                       [200, 50],
                       [70, 220]])

    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst


def perspective_transformation(img):
    rows, cols, ch = img.shape
    pts1 = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])
    pts2 = np.float32([[20, 30], [300, 0], [0, 300], [300, 300]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (cols, rows))
    return dst


def resize(img):
    scale_percent = 60
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def average_filters(img):
    dim = 2
    kernel = np.ones((dim, dim), np.float32) / dim*dim
    dst = cv2.filter2D(img, -1, kernel)
    return dst


def sharpen(img):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpen_img_1 = cv2.filter2D(img, -1, filter)
    return sharpen_img_1


def sharpen_laplacian(img):
    filter = np.array([[0,0,-1,0,0],[0,-1,-2,-1,0],[-1,-2,16,-2,-1],[0,-1,-2,-1,0],[0,0,-1,0,0]])
    mexican_hat_img1=cv2.filter2D(img,-1,filter)
    return mexican_hat_img1


def sharpen_flow(image_numbers):
    for img_nr in image_numbers:
        img = read_image(f'data/{img_nr}/{img_nr}.png')
        img = sharpen(img)
        cv2.imwrite(f'data/{img_nr}/-sharpen.png', img)
        extract_text_and_save(img, img_nr, "sharpen")

def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion


def dilation(img):
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    return dilation


def opening(img):
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening

def closing(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def morphological_gradient(img):
    kernel = np.ones((5, 5), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient


def generic_flow(image_numbers, flow_type):
    flow_name = flow_type.__name__
    for img_nr in image_numbers:
        img = read_image(f'data/{img_nr}/{img_nr}.png')
        img = flow_type(img)
        cv2.imwrite(f'data/{img_nr}/{flow_name}.png', img)
        extract_text_and_save(img, img_nr, f"{flow_name}")


flows = [normal,
         add_noise,
         affine_transformation, perspective_transformation,
         resize,
         average_filters,
         sharpen, sharpen_laplacian, erosion, dilation, opening, closing, morphological_gradient]

for flow in flows:
    generic_flow(images, flow)