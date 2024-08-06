import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from textblob import TextBlob
from colorama import Style, Fore

red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL

# 读取CSV文件
df = pd.read_csv('tolivecomment.csv', encoding='ANSI', header=None, names=['comment'])

# 定义情感分析函数
def analyze_sentiment(text):
    analysis = TextBlob(text)
    # 返回极性，范围从-1（负向）到1（正向）
    return analysis.sentiment.polarity

# 对每一行进行情感分析，并将结果存储在新的列中
df['sentiment'] = df['comment'].apply(analyze_sentiment)

# 根据情感值将评论分类
neutral_comments = df[df['sentiment'] == 0]['comment']
positive_comments = df[df['sentiment'] > 0]['comment']
negative_comments = df[df['sentiment'] < 0]['comment']

# 保存中性文本到CSV文件
neutral_comments.to_csv('neutral_comments.csv', index=False, header=False, encoding='utf-8')

# 保存正向文本到CSV文件
positive_comments.to_csv('positive_comments.csv', index=False, header=False, encoding='utf-8')

# 保存负向文本到CSV文件
negative_comments.to_csv('negative_comments.csv', index=False, header=False, encoding='utf-8')

# 计算平均情感极性
average_sentiment_neutral = df[df['sentiment'] == 0]['sentiment'].mean()
average_sentiment_positive = df[df['sentiment'] > 0]['sentiment'].mean()
average_sentiment_negative = df[df['sentiment'] < 0]['sentiment'].mean()

print(f"\n中性文本平均情感极性：{average_sentiment_neutral}")
print(f"正向文本平均情感极性：{average_sentiment_positive}")
print(f"负向文本平均情感极性：{average_sentiment_negative}")

# 输出结果
print(f"中性文本数量：{len(neutral_comments)}")
print(f"正向文本数量：{len(positive_comments)}")
print(f"负向文本数量：{len(negative_comments)}")

# 输出部分评论内容
print("\n中性文本：")
for comment in neutral_comments[:5]:
    print(comment)

print("\n正向文本：")
for comment in positive_comments[:5]:
    print(comment)

print("\n负向文本：")
for comment in negative_comments[:5]:
    print(comment)

# 绘制情感分布直方图
plt.figure(figsize=(12, 6))

# 主要y轴，绘制情感分布直方图
ax1 = sns.histplot(df['sentiment'], kde=True, color='skyblue')
ax1.set(xlabel='Sentiment Score', ylabel='Frequency')
ax1.set_title('Distribution of Sentiment Scores', fontsize=14, fontweight='bold', color='darkgreen')

# 次要y轴，绘制评论数量
ax2 = ax1.twinx()
ax2.scatter([average_sentiment_negative, average_sentiment_neutral, average_sentiment_positive], [len(negative_comments), len(neutral_comments), len(positive_comments)], color='red', marker='o')
ax2.set(ylabel='Comment Count')

# 添加文本注释
ax2.text(average_sentiment_negative, len(negative_comments), f'  Negative\n  {len(negative_comments)} comments\n  Avg Polarity: {average_sentiment_negative:.2f}', color='red', ha='right', va='bottom')
ax2.text(average_sentiment_neutral, len(neutral_comments), f'  Neutral\n  {len(neutral_comments)} comments\n  Avg Polarity: {average_sentiment_neutral:.2f}', color='red', ha='right', va='bottom')
ax2.text(average_sentiment_positive, len(positive_comments), f'  Positive\n  {len(positive_comments)} comments\n  Avg Polarity: {average_sentiment_positive:.2f}', color='red', ha='right', va='bottom')

plt.show()
