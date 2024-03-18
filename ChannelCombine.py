import os
import rasterio
from rasterio import merge


def main():
    # 调整为实际的文件夹路径
    band1_folder = 'data/train/filter/set2_0.01/band_1'
    band2_folder = 'data/train/filter/set2_0.01/band_2'
    output_folder = 'data/train/filter/set2_0.01/band_12'

    merge_two_bands_and_save(band1_folder, band2_folder, output_folder)


def merge_two_bands_and_save(band1_folder, band2_folder, output_folder):
    """
    合并两个波段的图像并保存为一个含两个波段的TIF图。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取第一个波段的所有图像文件
    band1_files = [os.path.join(band1_folder, f) for f in os.listdir(band1_folder) if f.endswith('.tif')]
    band1_files.sort()  # 确保文件是排序的

    # 对每个图像进行操作
    for band1_file in band1_files:
        file_name = os.path.basename(band1_file)
        file_name = file_name.replace("_band_1", "_band_2")
        # 构造第二个波段的对应图像文件路径
        band2_file = os.path.join(band2_folder, file_name)
        print(band2_file)
        if os.path.exists(band2_file):
            # 使用 rasterio 打开两个波段的图像
            with rasterio.open(band1_file) as src1, rasterio.open(band2_file) as src2:
                band1 = src1.read(1)
                band2 = src2.read(1)

                # 创建一个新的图像，其中包含两个波段
                profile = src1.profile
                profile.update(count=2)

                output_path = os.path.join(output_folder, file_name)
                output_path = output_path.replace("_band_2", "")
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(band1, 1)
                    dst.write(band2, 2)
                    print(f'Successfully saved {output_path}')
        else:
            print(f"Warning: Matching file for '{file_name}' not found in both folders.")



if __name__ == "__main__":
    main()