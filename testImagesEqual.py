import numpy as np
from PIL import Image


# Create Image object
noise = Image.open('testImagesPyTest/noise.png')
gt = Image.open('testImagesPyTest/gt.png')
should = Image.open('testImagesPyTest/shouldwork.png')


noise = np.divide(np.array(noise), 100.0)
print noise
gt = np.divide(np.array(gt), 100.0)
print gt
should = np.divide(np.array(should), 100.0)
print should

print "testing...\n\n\n"

print np.array(np.sum(np.abs(should - gt)))
print np.array(np.sum(np.abs(noise - gt)))

