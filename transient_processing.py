# -*- coding: utf-8 -*-
'''
create 2025-05-13 by wlf
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
from scipy import signal
import os
# 设置字体SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 带通滤波
def bandpass_filter(x, low_cut, high_cut, fs, order=8):
    """
    带通滤波器
    data: 输入信号
    lowcut: 低截止频率
    highcut: 高截止频率
    fs: 采样频率
    order: 滤波器阶数
    """
    sos = signal.butter(order, (low_cut, high_cut), btype="bandpass", output="sos", fs=fs)
    return signal.sosfiltfilt(sos, x, axis=0).astype(x.dtype)

# 陷波滤波
def notch_filter(x, exclude_freqs, fs, exclude_harmonics = False, max_harmonic = None, q = 30.0):
    def find_mutiples(base, limit):
        last_mult = int(limit / base)
        return [base * i for i in range(1, last_mult + 1)]
    
    # find harmonics if requrested
    if exclude_harmonics:
        if max_harmonic is None:
            max_harmonic = fs // 2
        exclude_freqs_set = set([f2 for f1 in exclude_freqs for f2 in find_mutiples(f1, max_harmonic)])
    else:
        exclude_freqs_set = set(exclude_freqs)

    for freq in exclude_freqs_set:
        b, a = signal.iirnotch(freq, q, fs)
        x = signal.filtfilt(b, a, x, axis=0).astype(x.dtype)

    return x

def load_hdsemg_signal(data_path):
    """
    读取数据
    data_path: 数据路径
    """
    # 读取数据
    fs = pd.read_csv(data_path, nrows=4, header=None)
    fs = float(list(fs.values)[1][0].split(':')[1])

    # 读取数据
    data = pd.read_csv(data_path, skiprows=4)
    n_samp = data.shape[0]
    n_ch = data.shape[1]-1
    index = np.arange(n_samp) / fs
    columns = [f'Ch{i}' for i in range(n_ch)]

    # pack in DataFrame
    emg = data[data.columns[list(range(64))]]
    emg.columns = columns

    return emg, fs

# 计算每个点的能量算子
def Teager_power_function(Signal):
    Tear_power = np.zeros(len(Signal))
    # 离散Teager能量算子的公式 = S(n)*S(n) -S(n-1) * S(n+1)
    for i in range(1, len(Signal) - 1):
        Tear_power[i] = abs(Signal[i]*Signal[i] - Signal[i-1]*Signal[i+1])  # 取绝对值得能量子
    return Tear_power

# 计算多通道的能量算子，并进行平滑
def channel_Teager(emg, window_size = 16):
    emg = emg - np.mean(emg) # 去均值化
    emg = emg / np.max(np.abs(emg)) # 归一化
    ch_n = emg.shape[1]
    sn = []
    window = np.ones(window_size) / window_size
    for i in range(ch_n):
        tke = Teager_power_function(emg[:, i])  # 每个通道的算子
        smoothed_tke = np.convolve(tke, window, mode='same')
        sn.append(smoothed_tke)
    return np.stack(sn, axis=1)

def tke_sliding_rms(sn, window_size=200, step = 20):
    rms_values = []
    for i in range(sn.shape[1]):
        for start in range(0, len(sn) - window_size, step):
            window = sn[start:start+window_size, i]
            rms = np.sqrt(np.mean(window**2))
            rms_values.append(rms)
    return np.array(rms_values)

def detect_transient_onsets(sn, fs, threshold_scale = 1.5, min_interval = 5.5, outlier_scale = 3.0, segment_length= 50, over_lap=0.1):
    """
    检测肌电信号起点
    参数：
    sn: ndarray 已经平滑的TKE能量子
    fs: int  采样率
    threshold_scale: float 动态阈值系数
    min_interval: float 相邻起点最小时间间隔 单位s
    outlier scale: float 异常值判断 高于均值+outlier_scale*std 的值被视为异常值
    segment_length: int 每个局部计算长度s(默认50s)
    over_lap: float, 重叠比例(0~1)

    返回：
        onset: list
        检测到的onset索引位置
    """

    min_interval_pts = int(min_interval * fs)
    segment_pts = int(segment_length * fs)
    step = int(segment_pts * (1 - over_lap))
    total_len = len(sn)

    onsets = []
    last_onset = -min_interval_pts
    for start in range(0, total_len, step):
        end = min(start + segment_pts, total_len)
        segment = sn[start:end]

        # 计算标准差
        mean_sn = np.mean(segment)
        print('mean_sn:', mean_sn)
        std_sn = np.std(segment)
        print('std_sn:', std_sn)
        # 加入异常值检测
        outlier_thresh = mean_sn + outlier_scale * std_sn
        print("outlier_thresh:", outlier_thresh)
        segment = np.where(segment > outlier_thresh, mean_sn, segment)
        print(outlier_thresh)

        # 设置动态阈值，进行能量检测
        threshold = mean_sn + threshold_scale * std_sn

        # 上升沿检测
        binary =(sn > threshold).astype(int)
        onset_raw =np.where(np.diff(binary) == 1)[0] + start

        # 最小间隔抑制
        for onset in onset_raw:
            if onset - last_onset > min_interval_pts:
                onsets.append(onset)
                last_onset = onset        

        if end == total_len:
            break

    return onsets

    
if __name__=='__main__':
    data_dir = "G:/recognition/subject/EMG_health"
    save_dir = "G:/recognition/prepocessed/transient/health"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    subjects = os.listdir(data_dir)
    
    for sub in subjects[1:2]:
        data_idx = []
        print(sub)
        save_path = os.path.join(save_dir, sub)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        sub_dir = os.path.join(data_dir, sub)

        datas = [i for i in os.listdir(sub_dir) if 'FilteredData' in i]
        print(datas)
        for data in datas[:]:
            data_dir_ = os.path.join(sub_dir, data)
            emg, fs = load_hdsemg_signal(data_dir_)
            print(emg.shape)
            ges_num = np.arange(int(data.split('-')[0]), int(data.split('-')[1]) + 1)
            emg = np.array(emg.values)
            emg = emg[int(fs):,:]
            print(f'ges_num:{ges_num}')
            emg = bandpass_filter(emg, 20, 500, fs, order=8)
            emg = notch_filter(emg, [200], fs, exclude_harmonics=True, q=35)
            sn = channel_Teager(emg)
            start_points = detect_transient_onsets(sn, fs=2000)
            print(len(start_points))
            # 绘图并标记起点
            plt.figure(figsize=(14, 10))
            plt.plot(sn, color='blue')
            if isinstance(start_points, list):
                for i, idx in enumerate(start_points):
                    plt.axvline(x=idx, color='r', linestyle='--', linewidth=1.5,
                                label='Detected Start' if i == 0 else "")
            else:
                plt.axvline(x=start_points, color='r', linestyle='--', linewidth=1.5, label='Detected Start')

            plt.title(f'TKE Signal with Detected Start - {data}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.legend()
            
            
            save = os.path.join(save_path, 'ges_' + data.split('-')[0] + '.png')
            plt.tight_layout()
            plt.savefig(save, dpi=300)
            plt.show()
            plt.close()



