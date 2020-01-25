import os
from glob import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class ImageResizer:

    def __init__(self):
        self.data_dir = '../data/img_align_celeba'
        self.IMAGE_HEIGHT = 28
        self.IMAGE_WIDTH = 28
        self.data_files = glob(os.path.join(self.data_dir, '*.jpg'))
        self.shape = len(self.data_files), self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3

    # Ostavimo samo lice na slici i stavimo na zeljenu velicinu
    def get_image(self, image_path, width, height, mode):
        image = Image.open(image_path)

        if image.size != (width, height):
            face_width = face_height = 108
            j = (image.size[0] - face_width) // 2
            i = (image.size[1] - face_height) // 2
            image = image.crop([j, i, j + face_width, i + face_height])
            image = image.resize([width, height], Image.BILINEAR)

        return np.array(image.convert(mode))

    def get_batch(self, images, width, height, mode='RGB'):
        data_batch = np.array(
            [self.get_image(tmp_image, width, height, mode) for tmp_image in images]).astype(np.float32)

        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))

        return data_batch

    def get_batches(self, batch_size):
        image_max_val = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = self.get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3])

            current_index += batch_size

            yield data_batch / image_max_val - 0.5


resizer = ImageResizer()
ims = resizer.get_batch(glob(os.path.join('../data/img_align_celeba', '*.jpg'))[:1], 56, 56)
# print(ims.shape)
# ims.shape = [1, 56, 56]
# print(ims.shape)

ims = ims[0]
print(ims / 255)
plt.imshow((ims * 255).astype(np.uint8))
plt.show()
print('e')


