

from Preprocess import split_train_val_oneband, create_datagen_oneband
from Unet_one_band import unet_one_band
from Loss import dice_loss,dice_coefficient,iou


from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from tensorflow.keras.optimizers import Adadelta




def main():
    # 参数设置
    batch_size = 2
    target_size = (256, 256)
    val_size = 0.2
    epochs = 50
    modelsavedir= 'unet_band1_val_best.hdf5'
    save_best_only = True
    image_folder = 'data/train/filter/set1_0.05/band_1'  # 图像文件夹路径
    mask_folder = 'data/train/filter/set1_0.05/labels'  # 掩模文件夹路径

    # 分割训练集和验证集
    train_images, train_masks, val_images, val_masks = split_train_val_oneband(image_folder, mask_folder, val_size=val_size)

    # 计算 steps_per_epoch 和 validation_steps
    steps_per_epoch = np.ceil(len(train_images) / batch_size)
    validation_steps = np.ceil(len(val_images) / batch_size)

    # 创建数据生成器
    train_gen = create_datagen_oneband(train_images, train_masks, batch_size, target_size)
    val_gen = create_datagen_oneband(val_images, val_masks, batch_size, target_size)

    # 创建模型
    model = unet_one_band()
    # 模型编译
    model.compile(optimizer=Adadelta(learning_rate=1.0, rho=0.95, epsilon=1e-8),
                  loss=dice_loss,
                  metrics=[dice_coefficient, iou])

    # 定义模型检查点回调
    model_checkpoint = ModelCheckpoint(modelsavedir, monitor='val_loss', verbose=1, save_best_only=save_best_only)

    # 模型训练
    model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[model_checkpoint],
        validation_data=val_gen,
        validation_steps=validation_steps
    )

if __name__ == "__main__":
    main()























