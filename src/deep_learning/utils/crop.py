import numpy as np

def crop_single_image(args):
    """
    Util function for each worker in a multiprocessing tool. Takes in an image and returns an cropped cells, cropped masks, and (y, x) center of 
    each cell
    """
    mask, image = args
    image_crops = []
    mask_crops = []
    centers = []
    for i in range(1, np.max(mask)):
        center = find_center(mask, i)
        if (center[0] < 38 or center[1] < 38 or
            center[0] > image.shape[0] - 38 or
            center[1] > image.shape[1] - 38):
            continue
        centers.append(center)
        crop = crop_img_from_center(center, image)
        crop = multiplex_mask_on_crop(crop, mask, i, center) # 75 * 75 * 4
        image_crops.append(crop)
        mask_crops.append(crop_mask_from_center(center, mask))
    return image_crops, mask_crops, centers

def crop_img_from_center(center, image):
    left = 0 # slighly assymetric, the left gets 38 pixels while the right gets 37 pixels
    right = 75
    bottom = 75
    top = 0
    if(center[0]>38): # Make sure h is not out of range
        if(center[0]<image.shape[0]-38):
            top = center[0] - 38
            bottom = center[0] + 37
        else:
            top = image.shape[0]-76
            bottom = image.shape[0]-1

    if(center[1]>38): # Make sure w is not out of range
        if(center[1]<image.shape[1]-38):
            left = center[1] - 38
            right = center[1] + 37
        else:
            left = image.shape[1]-76
            right = image.shape[1]-1
    
    return np.copy(image[top:bottom, left:right, :]) # For images

def multiplex_mask_on_crop(crop, mask, index, center): 

    for h in range(len(crop)): # O(n^2)
        for w in range(len(crop[0])):
            if(mask[h+center[0]-38, w+center[1]-38] != index) and (mask[h+center[0]-37, w+center[1]-37] != index): # there is a sight worry that i'm not matching the crop to mask pixel id perfectly, the extra if statement might be a temp fix 
                crop[h, w, :] = 0

    return crop

def crop_mask_from_center(center, image):
    left = 0 # slighly assymetric, the left gets 38 pixels while the right gets 37 pixels
    right = 75
    bottom = 75
    top = 0
    if(center[0]>38): # Make sure h is not out of range
        if(center[0]<image.shape[0]-38):
            top = center[0] - 38
            bottom = center[0] + 37
        else:
            top = image.shape[0]-76
            bottom = image.shape[0]-1 # making sure the bottom is not out of range

    if(center[1]>38): # Make sure w is not out of range
        if(center[1]<image.shape[1]-38):
            left = center[1] - 38
            right = center[1] + 37
        else:
            left = image.shape[1]-76
            right = image.shape[1]-1

    return np.copy(image[top:bottom, left:right]) # For masks

def find_center(mask, index):
    positions = np.argwhere(mask == index) # O(nlogn)

    if positions.size == 0:
        raise ValueError(f"No pixels found for index {index}")

    top = positions[:, 0].min()
    bottom = positions[:, 0].max()
    left = positions[:, 1].min()
    right = positions[:, 1].max()

    return int((top + bottom) / 2), int((left + right) / 2)
