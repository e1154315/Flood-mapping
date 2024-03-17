import numpy as np
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image_oneband(image):
    # 对数变换
    image = np.log1p(image)
    # 归一化到 [0, 1]
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image


def split_train_val_oneband(image_folder, mask_folder, val_size):
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
