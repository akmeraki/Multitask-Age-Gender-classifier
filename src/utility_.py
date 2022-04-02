import os 
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch 
import shutil
from collections import Counter
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

def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
	"""Save checkpoint if a new best is achieved
	
	state: checkpoint we want to save 
	is_best: if this checkpoint is the best so far
	checkpoint_path: path to save checkpoint
	best_model_path: path to save best model
	"""
	
	f_path = checkpoint_path

	# save checkpoint data to the path given, checkpoint_path
	torch.save(state, f_path)

	# if it is a best model, min validation loss
	if is_best:

		best_fpath = best_model_path
		# copy that checkpoint file to best path given, best_model_path

		shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    
	# initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    
	# initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    
	# initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    
	# return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()



def distribution_of_image_sizes(image_paths,root):

    new_paths = [os.path.join(root,x) for x in image_paths]
    
    img_shape_list = []
    for path in new_paths:
        img = Image.open(path).convert('RGB')
        img = np.asarray(img)
        img_shape_list.append(img.shape)

    # plt.figure(figsize=(14, 10))
    # plt.hist(img_shape_list,bins=5)
    # plt.title("Distribution of Image Sizes")
    count = Counter(img_shape_list)
    print(count)

# Show_img(gender_unknown_paths,4)
