import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件
df = pd.read_excel('merged_file.xlsx', sheet_name='Sheet1')

# 确保 'Time' 列存在，并将其转换为日期时间格式
df['Time'] = pd.to_datetime(df['Time'])

# 提取年份并创建 'Year' 列
df['Year'] = df['Time'].dt.year

# 按年份统计评论数量
yearly_comments = df.groupby('Year').size()

# 绘制折线图
plt.plot(yearly_comments.index, yearly_comments.values, marker='o')
plt.title('Number of Comments per Year')
plt.xlabel('Year')
plt.ylabel('Number of Comments')
plt.grid(True)

# 设置横坐标刻度为整数
plt.xticks(yearly_comments.index.astype(int))

plt.show()
