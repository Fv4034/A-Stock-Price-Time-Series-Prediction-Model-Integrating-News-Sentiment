import pandas as pd
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# 加载数据
file_path = 'C:/Users/WuJiaJun/anaconda3/sentiment/data/ExtractedNewsData.csv'
data = pd.read_csv(file_path)

# 移除缺失值
data = data.dropna(subset=['News'])

# 加载预训练的BERT模型和分词器
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name)

# 文本预处理函数
def preprocess_text(text):
    return tokenizer.encode(text, truncation=True, padding='max_length', max_length=128, return_tensors='tf')

# 预测函数
def predict_sentiment(text):
    input_ids = preprocess_text(text)
    output = model(input_ids)
    scores = tf.nn.softmax(output.logits, axis=-1)
    return scores.numpy().flatten()

# 计算情感分值
data['Sentiment_Scores'] = data['News'].apply(lambda x: predict_sentiment(x))

# 将情感分值添加到原始数据中
data.to_csv('C:/Users/WuJiaJun/anaconda3/sentiment/data/ProcessedNewsData_with_Sentiment.csv', index=False)
