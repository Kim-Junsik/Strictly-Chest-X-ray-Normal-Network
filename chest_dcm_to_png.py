import numpy as np
import pydicom
import skimage

def get_pixels_hu(path):
    dcm_image = pydicom.dcmread(path)
    try:
        image  = dcm_image.pixel_array
    except:
        print("Error == ", path)
    try:
        image  = apply_modality_lut(image, dcm_image)
    except:
        image = image.astype(np.int16)
        intercept = dcm_image.RescaleIntercept
        slope     = dcm_image.RescaleSlope
        if slope != 1:
            image = slope * image.astype(np.float64)
            image = image.astype(np.int16)
        image += np.int16(intercept)
    return np.array(image, dtype=np.int16)

def resize_and_padding_with_aspect_clahe(image, spatial_size):
    image = np.clip(image, a_min=np.percentile(image, 1.), a_max=np.percentile(image, 99.))
    image -= image.min()
    image /= image.max()                   
    image = skimage.img_as_ubyte(image)
    image = albu.PadIfNeeded(min_height=max(image.shape), min_width=max(image.shape), always_apply=True, border_mode=0)(image=image)['image']
    image = cv2.resize(image, spatial_size, interpolation=cv2.INTER_CUBIC)
    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
    image = skimage.util.img_as_float32(image)
    image = image * 255.
    return image

def dicom_to_png(input_img_path, img_size=(1024, 1024)):
    img = get_pixels_hu(input_img_path)
    img = resize_and_padding_with_aspect_clahe(img, img_size)
    return img