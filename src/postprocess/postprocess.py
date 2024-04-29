import cv2
import numpy as np
def postprocess(mask, guidance_mask):
    guidance_mask = cv2.imread(guidance_mask).astype(np.uint32)
    mask = cv2.imread(mask).astype(np.uint32)
    guidance_mask = cv2.imread(guidance_mask).astype(np.uint32)

    mask[:,:,0][mask[:,:,0]>100] = 225
    mask[:,:,0][mask[:,:,0]<=100] = 0

    mask[:,:,1][mask[:,:,1]>4] = 225
    mask[:,:,1][mask[:,:,1]<=4] = 0

    mask[:,:,2][mask[:,:,2]>100] = 225
    mask[:,:,2][mask[:,:,2]<=100] = 0

    # apply morphological operation on the guidance mask to fill the holes
    kernel = np.ones((21,21),np.uint8)

    guidance_mask = cv2.imread(guidance_mask).astype(np.uint32)
    guidance_mask = cv2.morphologyEx(guidance_mask, cv2.MORPH_CLOSE, kernel)
    

    mask[:,:,0] = mask[:,:,0] * (1-guidance_mask)
    mask[:,:,2] = mask[:,:,2] * guidance_mask

    return mask