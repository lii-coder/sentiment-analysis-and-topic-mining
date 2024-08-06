import matplotlib.pyplot as plt
import numpy as np
from gensim.models import LdaModel
import pandas as pd  # 处理数据
from gensim import corpora, models  # 创建字典和模型
from nltk.tokenize import word_tokenize  # 分词
from nltk.corpus import stopwords  # 停用词
from nltk.stem import WordNetLemmatizer  # 词性还原
import string
from gensim.models import CoherenceModel
import pyLDAvis.gensim_models

# 读取CSV文件
df = pd.read_csv('tolivecomment.csv', encoding='ANSI', header=None, names=['text'])

# 将文本分词并去除停用词和标点符号
stop_words = set(stopwords.words('english'))
translator = str.maketrans('', '', string.punctuation)
# nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # tokens = tokenizer.tokenize(word_tokenize(text.lower()))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]  # 去除非字母字符
    tokens = [word for word in tokens if word not in stop_words]  # 去除停用词
    tokens = [word.translate(translator) for word in tokens]  # 去除标点符号
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # 词性还原
    return tokens

# 预处理文本
df['tokens'] = df['text'].apply(preprocess_text)

# 保存处理后的文本为 txt 文件
# df['tokens'].apply(lambda tokens: ' '.join(tokens)).to_csv('processed_text.txt', index=False, header=None)

# 创建字典和文档词袋
dictionary = corpora.Dictionary(df['tokens'])
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# 准备存储主题数和困惑度的列表
num_topics_list = []
perplexity_list = []


# 遍历不同的主题数
for num_topics in range(1, 11):  # 假设尝试1到10个主题
    # 训练LDA模型
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    # 计算文档的困惑度
    perplexity = lda_model.log_perplexity(corpus)

    # 将结果添加到列表中
    num_topics_list.append(num_topics)
    perplexity_list.append(-perplexity)  # 取困惑度的负数

# 绘制图表
plt.plot(num_topics_list, perplexity_list, marker='o')
plt.title('Number of Topics vs. Perplexity')
plt.xlabel('Number of Topics')
plt.ylabel('Perplexity (Negative Log Scale)')
plt.show()
