from skimage import io, color
from PIL import Image

rgb = io.imread('test.jpg')
lab = color.rgb2lab(rgb)
print(lab.shape)
im = Image.fromarray(lab[:,:,0])
im.show()

minimum = [255, 255, 255]
maximum = [0, 0, 0]

for row in lab:
    for column in row:
        minimum[0] = min(minimum[0], column[0])
        minimum[1] = min(minimum[1], column[2])
        minimum[2] = min(minimum[2], column[1])

        maximum[0] = max(maximum[0], column[0])
        maximum[1] = max(maximum[1], column[2])
        maximum[2] = max(maximum[2], column[1])

print(minimum)
print(maximum)
