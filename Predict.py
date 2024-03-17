from Loss import dice_loss,dice_coefficient,iou
from Preprocess import preprocess_image_oneband
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
import os
import skimage.io as io
import zipfile


def main():
    # 加载模型时提供自定义损失函数的定义
    model = load_model('unet_band1_val_best.hdf5',
                       custom_objects={'dice_loss': dice_loss, 'dice_coefficient': dice_coefficient, 'iou': iou})

    # 指定要预测的图片目录和保存掩码的目录
    predict_dir = 'data/filter/image/band_1'
    save_dir = 'data/test_band_1'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    mask_value=0.5

    # 遍历目录中的所有图片，对每张图片进行预测并保存掩码
    for image_file in os.listdir(predict_dir):
        image_path = os.path.join(predict_dir, image_file)
        predict_oneband_save(model, image_path, target_size=(256, 256), original_size=(512, 512), mask_value=mask_value, save_path=save_dir)

    print("预测完成，掩码已保存至：", save_dir)

    # 指定目标目录和压缩文件名
    target_directory = "data/test_result"
    zip_file_name = "data/test_result/dhdata.zip"

    # 调用函数进行压缩
    zip_images(target_directory, zip_file_name)


def load_oneband_image(image_path, target_size=(256, 256)):
    """预处理图片：读取、缩放和归一化"""
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    preprocess_image_oneband(image)
    image = np.expand_dims(image, axis=0)  # 增加批次维度
    return image


def predict_oneband_save(model, image_path, target_size=(256, 256), original_size=(512, 512),mask_value=0.5,
                     save_path='data/predicted_masks'):
    """对单张图片进行预测，并保存掩码。现在会将预测的掩码调整为原始尺寸。"""
    image = load_oneband_image(image_path, target_size)
    mask = model.predict(image)[0]  # 预测并获取掩码
    mask = (mask > mask_value ).astype(np.float32)  # 二值化处理，这里改用 float32 以防在 resize 过程中出现问题

    # 将掩码调整为原始尺寸
    mask_resized = resize(mask, original_size, mode='constant', preserve_range=True)
    mask_resized = (mask_resized > mask_value).astype(np.uint8) * 255  # 再次二值化处理，确保掩码是黑白的

    file_name = os.path.basename(image_path)
    # 去掉文件名中的 '_band_1' 词缀，并确保保存为PNG格式
    file_name_without_band = file_name.replace('_band_1', '')
    mask_save_path = os.path.join(save_path, os.path.splitext(file_name_without_band)[0] + '_msk.png')

    io.imsave(mask_save_path, mask_resized.squeeze())  # 保存掩码图片

def zip_images(directory, zip_file_name):
    # 获取指定目录下所有文件的路径
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory)]

    # 初始化 zip 文件
    with zipfile.ZipFile(zip_file_name, 'w') as zipf:
        # 将目录下的所有文件添加到 zip 文件中
        for file_path in file_paths:
            zipf.write(file_path, os.path.basename(file_path))

if __name__ == "__main__":
    main()