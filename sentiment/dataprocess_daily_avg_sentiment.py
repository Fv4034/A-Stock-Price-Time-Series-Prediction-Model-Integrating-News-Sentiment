import pandas as pd
import ast

# 加载处理过的带有情感分数的数据
processed_file_path = 'C:/Users/WuJiaJun/anaconda3/sentiment/data/ProcessedNewsData_with_Sentiment.csv'
processed_data = pd.read_csv(processed_file_path)

# 函数：计算加权平均情感分数
def calculate_weighted_average(row):
    try:
        scores = ast.literal_eval(row['Sentiment_Scores'])
    except:
        # 手动添加逗号分隔符
        scores = list(map(float, row['Sentiment_Scores'][1:-1].split()))
    weighted_average = sum(score * (i + 1) for i, score in enumerate(scores))
    return weighted_average

# 计算每条新闻的加权平均情感分数
processed_data['Weighted_Average_Score'] = processed_data.apply(calculate_weighted_average, axis=1)

# 计算每天的平均情感得分
daily_average_sentiment = processed_data.groupby('Date')['Weighted_Average_Score'].mean().reset_index()

# 保存结果到新的CSV文件
output_path = 'C:/Users/WuJiaJun/anaconda3/sentiment/data/Daily_Average_Sentiment_Scores.csv'
daily_average_sentiment.to_csv(output_path, index=False)

print("每日平均情感得分已保存到文件：", output_path)
