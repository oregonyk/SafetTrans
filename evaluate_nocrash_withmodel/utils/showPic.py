import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集，假设数据集为CSV文件
data = pd.read_csv('../dataset/collision/CARLA_collision.csv')

# 获取属性值列
attribute_columns = data.iloc[:, :12]

# 获取标签列
label_columns = data.iloc[:, 13:]

# 绘制属性值的分布
plt.figure(figsize=(12, 6))
for i, col in enumerate(attribute_columns.columns):
    plt.subplot(3, 5, i + 1)
    plt.hist(attribute_columns[col], bins=20, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# 绘制标签的分布
plt.figure(figsize=(12, 3))
for i, col in enumerate(label_columns.columns):
    plt.subplot(2, 3, i + 1)
    plt.hist(label_columns[col], bins=20, alpha=0.7)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
