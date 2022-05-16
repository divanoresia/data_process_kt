import numpy as np
import pandas as pd
import torch
import csv
import pickle
from imageio import imread
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.nn as nn

#AI qixiang
# test version1
def image_read (fname,factor = 70):
    #雷达回波 factor = 70
    #风速  factor = 35
    #降水  factor = 10
    image = np.array(imread(fname)/255*factor)
    return image

def picture_idx_matrix(pic_path, dtype=np.str):
    pic_df = pd.read_csv(pic_path, header=None)  # header=None时，即指明原始文件数据没有列索引，这样read_csv会自动加上列索引，除非你给定列索引的名字。
    # pic_df = pic_df.map(lambda x: str(x)[:-4])# 最后一个字符
    # pic_df = pic_df[pic_df.str.contains('.png')]  # 这里填写需要删除的关键字！！！！！！！！！
    # for v in range(31217):
    #     #     temp = str(v).rjust(5,'0') + '.png'
    #     #     temp_end = str(v).rjust(5,'0')
    #     #     pic_df = pic_df.replace(to_replace=temp, value=temp_end)
    #     #     print(v)
    pic_idx_info = np.array(pic_df, dtype=dtype)
    return pic_idx_info


# def generate_dataset(seq_len, pre_len, time_len, split_ratio=0.8, normalize=True):
def generate_dataset_pre():
        all_data_precip,all_data_radar, all_data_wind = list(), list(), list()
        all_pic_idx_info = picture_idx_matrix(pic_path='/home/pesglab/home/lyj/kt/AI-QIXIANG/dataset_train_1.csv')
        order_value_i = all_pic_idx_info[:, 0]
        for i in range(10):
            # order_value_pd = pd.DataFrame(order_value_i)
            # temp_i = str(i).rjust(5,'0') + '.png'
            # temp_i_end = str(i).rjust(5,'0')
            # order_value_pd = order_value_pd.replace(to_replace=temp_i, value=temp_i_end)
            # order_value_array = order_value_pd.to_numpy()
            # data_precip = image_read(fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Precip/precip_' + str(i).rjust(5,'0') + '.png')
            # data_radar = image_read(fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Radar/radar_' + str(i).rjust(5,'0') + '.png')
            # data_wind = image_read(fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Wind/wind_' + str(i).rjust(5,'0') + '.png')
            temp_outside = str(order_value_i[i]).rjust(5,'0')
            temp_inside = temp_outside
            print(i)
            for u in range(41):
                data_precip = image_read(
                    fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Precip/precip_' + temp_inside.rjust(5,'0') + '.png')
                data_radar = image_read(
                    fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Radar/radar_' + temp_inside.rjust(5,'0') + '.png')
                data_wind = image_read(
                    fname='/home/pesglab/home/lyj/kt/AI-QIXIANG/Train/Wind/wind_' + temp_inside.rjust(5,'0') + '.png')

                all_data_precip.append(data_precip)
                all_data_radar.append(data_radar)
                all_data_wind.append(data_wind)
                temp_inside = str(int(temp_inside) + 1)

        return np.array(all_data_precip),np.array(all_data_radar),np.array(all_data_wind)


def generate_dataset(
                all_data_precip,all_data_radar,all_data_wind, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
        ):

        if time_len is None:
            time_len = all_data_precip.shape[0]
        if normalize:
            max_val_precip = np.max(all_data_precip)
            data_precip_nor = all_data_precip / max_val_precip
            max_val_radar = np.max(all_data_radar)
            data_radar_nor = all_data_radar / max_val_radar
            max_val_wind = np.max(all_data_wind)
            data_wind_nor = all_data_wind / max_val_wind
        train_size = int(time_len * split_ratio)
        train_data_precip = data_precip_nor[:train_size]
        train_data_radar = data_radar_nor[:train_size]
        train_data_wind = data_wind_nor[:train_size]
        test_data_precip = data_precip_nor[train_size:time_len]
        test_data_radar = data_radar_nor[train_size:time_len]
        test_data_wind = data_wind_nor[train_size:time_len]
        train_length,_1,_2 = train_data_precip.shape
        test_length,_3,_4 = test_data_precip.shape
        train_precip, train_radar, train_wind, test_precip, test_radar, test_wind = list(), list(), list(), list(), list(), list()

        for i in range(int(train_length) - seq_len - pre_len):
            train_precip.append(np.array(train_data_precip[i: i + seq_len]))
            train_radar.append(np.array(train_data_radar[i: i + seq_len]))
            train_wind.append(np.array(train_data_wind[i: i + seq_len]))

        for i in range(int(test_length) - seq_len - pre_len):
            test_precip.append(np.array(test_data_precip[i: i + seq_len]))
            test_radar.append(np.array(test_data_radar[i: i + seq_len]))
            test_wind.append(np.array(test_data_wind[i: i + seq_len]))
        return np.array(train_precip), np.array(train_radar), np.array(train_wind), np.array(test_precip), np.array(test_radar), np.array(
            test_wind),max_val_precip,max_val_radar,max_val_wind


def generate_torch_datasets(
    seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    #pre_processing_precip.shape = (410,480,560)
    pre_processing_precip, pre_processing_radar, pre_processing_wind = generate_dataset_pre()
    train_precip, train_radar, train_wind, test_precip, test_radar, test_wind,max_val_precip,max_val_radar,max_val_wind= generate_dataset(
        pre_processing_precip,
        pre_processing_radar,
        pre_processing_wind,
        #seq_len,
        seq_len,
        #pre_len,
        pre_len,
        time_len=time_len,
        split_ratio=split_ratio,
        normalize=normalize,
    )
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_precip), torch.FloatTensor(train_radar),torch.FloatTensor(train_wind)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_precip), torch.FloatTensor(test_radar), torch.FloatTensor(test_wind)
    )
    return train_dataset, test_dataset