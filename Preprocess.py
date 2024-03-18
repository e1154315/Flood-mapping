import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
import rasterio  # 新添加，用于读取tif文件
import skimage.transform as trans


def preprocess_image_oneband(image):
    # 对数变换
    image = np.log1p(image)
    # 归一化到 [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def preprocess_image_twoband(image_path, target_size=(256, 256)):
    """针对双波段tif影像的预处理函数"""
    with rasterio.open(image_path) as src:
        image = src.read([1, 2])  # 读取前两个波段
        image = np.moveaxis(image, 0, -1)  # 重排轴，从(channels, height, width)到(height, width, channels)
        image = trans.resize(image, target_size)  # 调整大小
        image = np.log1p(image)  # 对数变换
        image = (image - image.min()) / (image.max() - image.min())  # 归一化到[0,1]
    return image

def preprocess_mask(mask_path, target_size=(256, 256)):
    """掩码图像的预处理函数，不进行归一化"""
    mask = load_img(mask_path, color_mode='grayscale', target_size=target_size)
    mask = img_to_array(mask)
    return mask
# 这里省略了split_train_val和其它配置代码，因为它们不需要修改

def split_train_val(image_folder, mask_folder, val_size):
    image_paths = glob.glob(os.path.join(image_folder, '*.tif'))  # 假设使用png格式，根据实际情况修改
    mask_paths = glob.glob(os.path.join(mask_folder, '*.png'))  # 假设掩模也是png格式

    # 确保图像和掩模是匹配的
    image_paths.sort()
    mask_paths.sort()

    # 划分训练集和验证集
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=val_size, random_state=42
    )

    return train_images, train_masks, val_images, val_masks





def create_datagen_oneband(image_paths, mask_paths, batch_size, target_size):
    # 将图像和掩码路径列表转换为 pandas DataFrame
    image_df = pd.DataFrame({'filename': image_paths})
    mask_df = pd.DataFrame({'filename': mask_paths})

    image_datagen = ImageDataGenerator(preprocessing_function=preprocess_image_oneband)
    mask_datagen = ImageDataGenerator()

    # 使用 DataFrame 代替列表
    image_generator = image_datagen.flow_from_dataframe(
        dataframe=image_df,
        x_col='filename',
        y_col=None,
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe=mask_df,
        x_col='filename',
        y_col=None,
        class_mode=None,
        color_mode='grayscale',
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False
    )

    while True:
        img = next(image_generator)
        mask = next(mask_generator)
        yield img, mask
class TwoBandDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, target_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.array([preprocess_image_twoband(image_path, self.target_size) for image_path in batch_images])
        masks = np.array([preprocess_mask(mask_path, self.target_size) for mask_path in batch_masks])

        return images, masks
