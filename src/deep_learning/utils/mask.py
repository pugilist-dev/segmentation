import numpy as np

def binary_masks(masks):
    """
    Takes an np.array (shape N, 75, 75) with instance values and converts it to a np.array (shape N, 1, 75, 75)
    with 1 or 0 in the second dimension to indicate mask or no mask.
    Any nonzero value in the original mask is set to 1.
    """
    # Ensure input is a numpy array
    masks = np.asarray(masks)
    # Create binary masks: 1 where mask > 0, else 0
    binary = (masks > 0).astype(np.uint8)
    # Add a channel dimension (axis=1)
    binary = binary[:, np.newaxis, :, :]
    return binary
