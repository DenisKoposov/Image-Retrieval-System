import cv2
from matplotlib import pyplot as plt
from Utils import filter_points

def draw(patches, labels, numbers, nc=4):

    nr = len(patches) // nc
    if len(patches) % nc:
        nr += 1

    plt.figure(figsize=(10, 8))
    ax = []

    for i in range(len(patches)):
        ax.append(plt.subplot(nr, nc, i+1))
        plt.title(labels[i] + " " + str(numbers[i]))

    for i, a in enumerate(ax):
        a.axis('off')
        a.imshow(patches[i])

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()

def detect(detector, image, max_keypoints=None, out=None):
    '''
    Detector object must have method 'detect()'
    '''
    kp = detector.detect(image)
    kp = filter_points(kp, 10)
    return sorted(kp,
                  key=lambda x: x.response,
                  reverse=True)[:max_keypoints]

detectors = {
             #'mser': cv2.MSER_create(),
             'sift': cv2.xfeatures2d.SIFT_create(),
             'surf': cv2.xfeatures2d.SURF_create(400),
             #'brisk': cv2.BRISK_create(45),
             #'sblob': cv2.SimpleBlobDetector_create(),
             #'star': cv2.xfeatures2d.StarDetector_create()
             }

TEST_IMAGE_PATH = '/media/kopoden/DATA/shared/NIR/code/photo_archive/oxford/5k/worcester_000137.jpg'
max_keypoints = 300

img = cv2.imread(TEST_IMAGE_PATH, 0)
print("Type of image:", type(img))
img_draw = cv2.imread(TEST_IMAGE_PATH, cv2.IMREAD_ANYCOLOR)
key_points = [ (detect(d, img, max_keypoints=max_keypoints), l) for l, d in detectors.items() ]

key_points, labels = list(zip(*key_points))
results = [ cv2.drawKeypoints(img_draw, kp, None) for kp in key_points ]
numbers = [ len(kp) for kp in key_points]
draw(results, labels, numbers, nc=5)