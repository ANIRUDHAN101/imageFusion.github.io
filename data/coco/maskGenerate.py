# %%
import cv2
from pycocotools.coco import COCO
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
#%%
coco = COCO('/home/anirudhan/project/image-fusion/data/coco/annotations/instances_train2017.json')
img_dir = '/home/anirudhan/project/image-fusion/data/coco/images/train2017'
# %%
def coco2Mask(img):
    def cmpKey(x):
        return x['area']
    
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    if anns:
        anns.sort(key=cmpKey, reverse=True)
        anns_image = np.zeros((img['height'], img['width']))
        anns_image = np.maximum(anns_image, coco.annToMask(anns[0])*anns[0]['category_id'])
        #convert it into binary image
        binary_mask = (anns_image > 0).astype(np.uint8)
    # Replace 'gray_image.jpg' with the desired output file pathmask_img_dir = '/home/pn_kumar/fusion/GACN/data/myd ata/mask2017_val'

        return True, binary_mask*255
    return False, None
# %%
# load all the images from coco training set and save them as binary masks
import os
import cv2
import concurrent.futures

# Define a function to process an image
def process_image(image_id, coco, mask_img_dir):
    img = coco.imgs[image_id]
    take, mask = coco2Mask(img)
    if take:
        print(f"Processing image {image_id}")
        # save the images usning cv2 by name having 12 digits with leading zeros
        cv2.imwrite(os.path.join(mask_img_dir, str(image_id).zfill(12) + '.jpg'), mask)

# Define the directory where masks will be saved
# mask_img_dir = '/home/pn_kumar/fusion/GACN/data/mydata/mask2017_val'
mask_img_dir = '/home/anirudhan/project/image-fusion/data/coco/images/train_2017mask'

# Get the list of image IDs
images = coco.getImgIds()

# Define the number of threads you want to use
num_threads = 16  # Adjust this number according to your needs

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the thread pool
    futures = [executor.submit(process_image, image_id, coco, mask_img_dir) for image_id in images]

    # Wait for all tasks to complete
    concurrent.futures.wait(futures)

print("All threads have finished processing.")

# %%
# import os
# dataDir = "/home/pn_kumar/fusion/GACN/data/mydata/train2017"
# maskDier = "/home/pn_kumar/fusion/GACN/data/mydata/mask2017_train"
# imgNames = os.listdir(dataDir)
# imgMasks = os.listdir(maskDier)
# # %%
# print(imgNmes)
# # %%
# len(imgNames)
# # %%
