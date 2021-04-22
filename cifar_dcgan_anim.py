#pip install -q imageio
import imageio
#import PIL
import glob
from PIL import Image

import os

cwd = os.getcwd()
files = os.listdir(cwd)

# Display a single image using the epoch number
def display_image(epoch_no):
    return Image.open(r'C:\Users\User\Documents\virtual\cifar_dcgan4\image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(20)

# ANIMATION
anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
    filenames = glob.glob(r'C:\Users\User\Documents\virtual\cifar_dcgan4\image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
    image = imageio.imread(filename)
    writer.append_data(image)

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)
