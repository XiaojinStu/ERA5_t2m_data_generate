import xarray as xr
import numpy as np
import random
import json
import os
from tqdm import tqdm

# 定义数据集路径
dataset_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr/'

# 打开ERA5数据集
ds = xr.open_zarr(dataset_path)

# 提取2米温度数据并采样为8小时间隔
t2m = ds['2m_temperature'].resample(time='8H').nearest()

# 数据预处理函数
def preprocess_sample(t2m_sample):
    # 归一化温度
    mean = np.mean(t2m_sample)
    std = np.std(t2m_sample)
    t2m_sample = (t2m_sample - mean) / std

    # 提取纬度和经度信息
    latitudes = np.sin(np.deg2rad(t2m_sample.latitude))
    longitudes_sin = np.sin(np.deg2rad(t2m_sample.longitude))
    longitudes_cos = np.cos(np.deg2rad(t2m_sample.longitude))

    # 提取时间信息
    time = t2m_sample.time
    day_of_year = time.dt.dayofyear
    time_of_day = time.dt.hour + time.dt.minute / 60.0
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.0)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.0)
    time_of_day_sin = np.sin(2 * np.pi * time_of_day / 24.0)
    time_of_day_cos = np.cos(2 * np.pi * time_of_day / 24.0)

    # 返回处理后的样本
    return {
        't2m': t2m_sample.values,
        'latitudes': latitudes.values,
        'longitudes_sin': longitudes_sin.values,
        'longitudes_cos': longitudes_cos.values,
        'day_of_year_sin': day_of_year_sin.values,
        'day_of_year_cos': day_of_year_cos.values,
        'time_of_day_sin': time_of_day_sin.values,
        'time_of_day_cos': time_of_day_cos.values
    }

# 将样本转换为指定的JSON格式
def format_sample(sample):
    coords = [[lat, lon_sin, lon_cos] for lat, lon_sin, lon_cos in zip(sample['latitudes'], sample['longitudes_sin'], sample['longitudes_cos'])]
    start_time_encoding = [
        sample['day_of_year_sin'][0],
        sample['day_of_year_cos'][0],
        sample['time_of_day_sin'][0],
        sample['time_of_day_cos'][0]
    ]
    data = sample['t2m'].tolist()

    formatted_sample = {
        'description': {
            'coords': coords,
            'start': start_time_encoding
        },
        'data': data
    }
    return formatted_sample

# 生成样本数据
def generate_samples(t2m, num_samples):
    samples = []
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        # 随机选择时间点和空间网格点
        time_index = random.randint(0, len(t2m.time) - 16) #start
        lat_lon_indices = random.sample(range(t2m.shape[1] * t2m.shape[2]), random.randint(60, 90))#这里数据集只有64*32 = 2048,  xVal里面说是100万空间网格，最后采样对齐60-90
        time_points = slice(time_index, time_index + random.randint(8, 16))   #include 8–16 temperature time points at 8-hour intervals (corresponding to 2–4 days)

        # 提取样本并预处理
        t2m_sample = t2m.isel(time=time_points).stack(grid=('latitude', 'longitude')).isel(grid=lat_lon_indices)
        sample = preprocess_sample(t2m_sample)
        formatted_sample = format_sample(sample)
        samples.append(formatted_sample)

    return samples

# 定义保存路径
output_dir = 'ERA5_t2m_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取输入的样本数量
num_samples = int(input("请输入要生成的样本数量: "))

# 生成完整的数据集
all_samples = generate_samples(t2m, num_samples=num_samples)

# 将样本保存为JSON文件
output_file = os.path.join(output_dir, 'all_samples.json')
with open(output_file, 'w') as f:
    json.dump(all_samples, f, indent=4)  # 使用 indent 参数进行格式化

print("数据集生成完成并已保存为JSON文件")
