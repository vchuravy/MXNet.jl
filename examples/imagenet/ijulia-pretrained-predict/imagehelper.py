import numpy as np
from skimage import io, transform

def PreprocessImage(img):
    img = np.array(img)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 256

    #-------------------------------------------------------------------
    # Note: The decoded image should be in BGR channel (opencv output)
    # For RGB output such as from skimage, we need to convert it to BGR
    # WRONG channel will lead to WRONG result
    #-------------------------------------------------------------------
    # swap channel from RGB to BGR
    sample = sample[:, :, [2,1,0]]
    # swap axes to make image from (224, 224, 4) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)

    sample.resize(3,224,224)
    return sample
