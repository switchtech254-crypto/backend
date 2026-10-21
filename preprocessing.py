#preprocessing.py
#convert the captured images to grayscale
#determine whether the background is light or dark
#seperate the digit from the background using thresholding
#crop around the detected digit
#resize it proportionally to fit 20 X 20 pixels
#center it on a 28 X 28 black canvas and normalize to [0,1]
from io import BytesIO #read image bytes in memory
import numpy as np 
from PIL import Image, ImageOps#pillow
#Image opens images, creates canvases, crops images, resizes them and pastes image regions
#ImageOps provides additional image-preprocessing operations including automatic contrast adjustmet

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """convert a captured handwritten digit into an MNIST-like input""" #docstring

    #decode the uploaded image and convert to grayscale
    image = Image.open(BytesIO(image_bytes)).convert("L")
    #Image.open() decodes the image and creates a pillow image object
    #BytesIO() prepresents the uploaded bytes as a file-like object
    #"L" mode means 8 bit gray scale (turn the image from colored to black and white)
    image = ImageOps.autocontrast(image)#improve contrast

    gray = np.asarray(image, dtype=np.uint8)#convert the image into a numpy array

    #estimate background brightness using the image borders
    border = np.concatenate([
        gray[0, :],
        gray[-1,:],
        gray[:, 0],
        gray[:,-1],
    ])

    #determine whether the background is light
    background_is_light = float(np.median(border)) > 127 #does it exceed 127? if it does background_is_light becomes True
    #127 is approximately the midpoint of 0-255 intensity range

    #Otsu thresholding: find a threshold from the image histogram
    #build the image histogram
    #this creates a histogram of pixel brightness
    #gray.ravel() flattens the 2D pixel matrix into a 1D array
    #np.bincount() counts how often each integer occurs eg. intesity 0 occurs once
    histogram = np.bincount(gray.ravel(), minlength=256)
    total = gray.size #total number of the pixels. for an image with 480 rows and 640 columns total = 480x640=307200
    sum_total = np.dot(np.arange(256), histogram)#sum all pixel totals. histogram contains the count of each intensity
    #np.dot() multiplies each intensity by its count and adds all the products together

    #initialize the Otsu thresholding variables
    weight_background = 0
    sum_background = 0 
    best_variance = -1.0
    threshold = 127

    #search all possible thresholds
    #this loop tries threshold candidates from 0 through 255
    #for every candidate t, the algorithm calculates how well the image's intensity distribution
    #seperates into 2 classes.
    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]

        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance = (weight_background * weight_foreground * (mean_background - mean_foreground)**2)

        if variance > best_variance:
            best_variance = variance
            threshold = t

    #make the digit white and the background black, like MNIST
    #the purpose is to create a binary mask: an array of True and False values
    #if the background is light, the dark pixels are treated as the digit
    #if the background is dark, the light pixels are treated as the digit
    if background_is_light:
        mask = gray < threshold
    else:
        mask = gray > threshold
    
    #ignore isolated noise by finding the foreground bounding box
    #find the digit's bounding box
    ys, xs = np.where(mask)
    
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No digit detected. Try better lighting and dark handwriting")
    
    #calculate the bounding box coordinates
    #these 4 values identify the smallest rectangular region containing every detected foreground pixel
    #xs.min() finds the leftmost foreground column
    #xs.max() finds the rightmost foreground column
    #ys.min() finds the topmost foreground row
    #ys.max() finds the bottommost foreground row
    # +1 is necessary since numpy slices exclude their ending index eg values=[2,3,4,5] values[1:4] gives [2,3]
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1
    
    #crop the digit
    #mask[y_min:y_max, x_min:x_max] extracts the bounding box region
    #astype(np.uint8)converts the boolean mask to integers. False becomes 0, True becomes 1
    # *255 converts those values to standard black-and-white pixel intensities. 0 becomes black, 1 becomes white or 255
    #the resulting digit tightly cropped, binary image with no unnecessary outer background  
    digit = mask[y_min:y_max, x_min:x_max].astype(np.uint8) * 255
    
    #keep the digit's aspect ratio and fit it inside a 20 x 20 region
    digit_image = Image.fromarray(digit)#Image.fromarray() converts the image array into a pillow image so that you can use pillow's resizing options
    digit_image.thumbnail((20, 20), Image.Resampling.LANCZOS) #Image.Resampling.LANCZOS specifies a high-quality resampling filter used during resizing.
    
    #center it on a 28 x 28 black canvas
    canvas = Image.new("L", (28,28), color=0)
    
    x_offset = (28 - digit_image.width) // 2
    y_offset = (29 - digit_image.height) // 2
    
    canvas.paste(digit_image, (x_offset, y_offset))
    
    #shape: (1, 28, 28, 1), matching the cnn input
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array[np.newaxis, ..., np.newaxis]
        