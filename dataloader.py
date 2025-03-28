import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class StockDataset(Dataset):
    def __init__(self, dataPath, window, is_test=False):
        df1 = pd.read_csv(dataPath)
        df1 = df1.iloc[:, 1:]  # 保留除了日期外的所有列
        
        # 分离Sentiment列
        sentiment = df1['Sentiment'].values.reshape(-1, 1)
        other_features = df1.drop(columns=['Sentiment'])
        
        # 对其他列进行归一化处理
        scaler = MinMaxScaler()
        other_features_scaled = scaler.fit_transform(other_features)
        
        # 对Sentiment列单独进行归一化处理
        sentiment_scaler = MinMaxScaler()
        sentiment_scaled = sentiment_scaler.fit_transform(sentiment)
        
        # 将缩放后的Sentiment列添加回去
        df_scaled = pd.DataFrame(other_features_scaled, columns=other_features.columns)
        df_scaled['Sentiment'] = sentiment_scaled.flatten()

        # 将Sentiment列中的nan值替换为0.5
        df_scaled['Sentiment'].fillna(0.5, inplace=True)

        input_size = len(df_scaled.iloc[1, :])
        stock = df_scaled
        seq_len = window
        amount_of_features = len(stock.columns)  # 有几列
        data = stock.values  # 表格转化为矩阵
        sequence_length = seq_len + 1  # 序列长度
        result = []
        for index in range(len(data) - sequence_length):  # 循环数据长度-sequence_length次
            result.append(data[index: index + sequence_length])  # 第i行到i+sequence_length
        result = np.array(result)  # 得到样本，样本形式为6天*特征数
        row = round(0.9 * result.shape[0])  # 划分训练集测试集
        train = result[:int(row), :]
        x_train = train[:, :-1]
        y_train = train[:, -1][:, -4]  # 假设y_train是收盘价
        x_test = result[int(row):, :-1]
        y_test = result[int(row):, -1][:, -4]
        # reshape成 时间步长*特征数
        X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
        X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  
        if not is_test:
            self.data = X_train
            self.label = y_train
        else:
            self.data = X_test
            self.label = y_test
            
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        return torch.from_numpy(self.data[idx]).to(torch.float32), torch.FloatTensor([self.label[idx]])


