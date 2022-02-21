import os 
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid

# Visualize the images in a Grid 
def show_img(gender_unknown_paths,number_to_display):

    paths = np.random.choice(gender_unknown_paths,number_to_display)
    
    root = os.getcwd()    
    new_paths = [os.path.join(root,'data','aligned',x) for x in paths]

    img_list = []

    for path in new_paths:
        img = Image.open(path).convert('RGB')
        # img = img.resize((150,150))
        img = np.asarray(img)
        img_list.append(img)

    fig = plt.figure(figsize=(10, 10))
    grid = ImageGrid(fig, nrows_ncols=(int(number_to_display/2), int(number_to_display/2)),rect=111)

    for ax, im in zip(grid, img_list):
        ax.imshow(im)   

    plt.show()     

# show_img(gender_unknown_paths,4)