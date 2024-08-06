import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pyLDAvis.gensim_models


# 数据加载
df = pd.read_csv('tolivecomment.csv', encoding='ANSI', header=None, names=['comment'])

# 数据预处理
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Additional custom lemmatization rules
custom_lemmatization_rules = {
    'amazing': 'amaze',  # Add your custom rules here
    'emotional': 'emotion',
    'emotionally': 'emotion',
    'historical': 'history',
    'historically': 'history',
    'depressing': 'depress',
    'culturally': 'culture',
    'translator': 'translation',
    'translate': 'translation',
    'recommendation': 'recommend',
    'recommendable': 'recommend',
    'straightforwardly': 'straightforward',
    'straightforwardness': 'straightforward',
    'written': 'writing',
    'touching': 'touch',
    'suffering': 'suffer',
    'beautifully': 'beautiful',
    'sadness': 'sad',
    'die': 'death',
    'hopeful': 'hope',
    'powerful': 'power',
    'hardship': 'suffer',
    'simplistic': 'simple',
    'tragic': 'tragedy'
    # Add more rules as needed
}
def preprocess_text(text):
    # Tokenize and tag POS
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    # Lemmatize based on POS with custom rules
    words = [custom_lemmatization_rules.get(word.lower(), lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(pos_tag)))
             for word, pos_tag in pos_tags
             if word.isalpha() and word.lower() not in stop_words]

    return words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return 'a'  # Adjective
    elif treebank_tag.startswith('V'):
        return 'v'  # Verb
    elif treebank_tag.startswith('N'):
        return 'n'  # Noun
    elif treebank_tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to Noun

df['preprocessed_comment'] = df['comment'].apply(preprocess_text)

# Save the preprocessed data to an Excel file
# output_excel_path = 'preprocessed_data.xlsx'
# df.to_excel(output_excel_path, index=False, float_format='%.2f')  # Use float_format if needed

# print(f"Preprocessed data saved to {output_excel_path}")

# 创建语料库和字典
dictionary = corpora.Dictionary(df['preprocessed_comment'])
corpus = [dictionary.doc2bow(text) for text in df['preprocessed_comment']]

# 使用TF-IDF进行特征提取
tfidf_model = models.TfidfModel(corpus, id2word=dictionary, normalize=True, smartirs='ntc', pivot=1.0, slope=0.25)
tfidf_corpus = tfidf_model[corpus]

lda_model = LdaModel(tfidf_corpus, num_topics=3, id2word=dictionary, alpha='auto', eta='auto', iterations=100, random_state= 15)

# 打印每个主题的关键词
num_top_words = 20
for topic_idx in range(3):
    top_keywords = lda_model.show_topic(topic_idx, topn=num_top_words)
    top_keywords = [word for word, _ in top_keywords]
    print(f"Topic #{topic_idx + 1}: {', '.join(top_keywords)}")

# 获取每个文档的主题分布
topics_distribution = lda_model[tfidf_corpus]

# 可视化
vis = pyLDAvis.gensim_models.prepare(lda_model, tfidf_corpus, dictionary, sort_topics=False)
pyLDAvis.save_html(vis, 'lda_visualization_gensim.html')

# 创建一个字典来存储每个主题的文档数量和文档内容
topic_documents = {i: [] for i in range(lda_model.num_topics)}

# 遍历每个文档，将其添加到相应主题的列表中
for doc_id, doc_topics in enumerate(topics_distribution):
    dominant_topic = max(doc_topics, key=lambda x: x[1])[0] if doc_topics else None
    topic_documents[dominant_topic].append(df.iloc[doc_id]['comment'])

# 将每个主题下的文档保存为CSV文件
for topic, documents in topic_documents.items():
    topic_df = pd.DataFrame({'comment': documents})
    topic_df.to_csv(f'topic_{topic + 1}_documents.csv', index=False)

# 打印每个主题的文档数量和占比
total_documents = len(df)
for topic, documents in topic_documents.items():
    count = len(documents)
    percentage = (count / total_documents) * 100
    print(f"Topic #{topic + 1}: Documents={count}, Percentage={percentage:.2f}%")


