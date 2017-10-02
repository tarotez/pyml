import os
from PIL import Image

# parameters
new_im_width = 32
new_size = (new_im_width, new_im_width);
# input_dir = '../data/shiba_resized'
# output_dir = '../data/class_0'
input_dir = '../data/tezu_resized'
output_dir = '../data/class_1'

# open directory
imagefiles = os.listdir(input_dir)

i = 0
for filename in imagefiles:
	if not filename.startswith("."):
		input_path = input_dir + "/" + filename
		output_path = output_dir + "/" + str(i) + ".jpg"
		img = Image.open(input_path,'r')
		img.thumbnail(new_size, Image.ANTIALIAS)
		img.save(output_path, 'JPEG', quality=100, optimize=True)
		i += 1

