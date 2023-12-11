from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import numpy as np


COLOURS = {
    'red': [255, 0, 0],
    'red-orange': [255, 69, 0],
    'orange': [255, 165, 0],
    'yellow-orange': [255, 215, 0],
    'yellow': [255, 255, 0],
    'yellow-green': [173, 255, 47],
    'green': [0, 128, 0],
    'blue-green': [0, 255, 127],
    'blue': [0, 0, 255],
    'indigo': [75, 0, 130],
    'violet': [238, 130, 238],
    'purple': [128, 0, 128],
    'magenta': [255, 0, 255],
    'pink': [255, 192, 203],
    'cyan': [0, 255, 255],
    'turquoise': [64, 224, 208],
    'sky-blue': [135, 206, 235],
    'lime-green': [50, 205, 50]
}

def find_closest(pixel):
    min_distance = float('inf')
    closest = None
    for colour_name, colour in COLOURS.items():
        distance = np.linalg.norm(np.array(pixel) - np.array(colour))
        if distance < min_distance:
            min_distance = distance
            closest = colour_name
    return COLOURS[closest]

img = Image.open("data/large_orange_lake.jpg")
img_np = np.array(img)

reshaped_img = img_np.reshape((-1, 3))

# change this value based on complexity of the image
num_clusters = 9

kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto')
kmeans.fit(reshaped_img)
labels = kmeans.predict(reshaped_img)
cluster_centers = kmeans.cluster_centers_
segmented_img = cluster_centers[labels].reshape(img_np.shape).astype(np.uint8)

solid_lines_img = Image.new('RGB', img.size, (255, 255, 255))
draw = ImageDraw.Draw(solid_lines_img)

# Draw solid lines between clusters
for y in range(img.height - 1):
    for x in range(img.width - 1):
        if labels[y * img.width + x] != labels[y * img.width + x + 1] or \
           labels[y * img.width + x] != labels[(y + 1) * img.width + x]:
            draw.line([(x, y), (x + 1, y)], fill='black', width=1)
            draw.line([(x, y), (x, y + 1)], fill='black', width=1)

solid_lines_img.save('data/picture_clustered.jpg')