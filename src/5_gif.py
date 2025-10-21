#!/usr/bin/env python
# coding: utf-8

# In[3]:


import glob
import imageio

image_folder = '../pics/'
images = [image_folder + str(i) + ".png" for i in range(1,34)]

images[0:0] = [images[0]]
images[0:0] = [images[0]]

image_list = []
for image_path in images:
    image = imageio.imread(image_path)
    image_list.append(image)

gif_image_path = '../pics/output.gif'
imageio.mimsave(gif_image_path, image_list, duration=0.5)


# In[ ]:




