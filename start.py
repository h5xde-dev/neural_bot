import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve

import requests

import re
import pymorphy2
from collections import Counter
from wordcloud import WordCloud
from tqdm import tqdm

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

toxic_comments = pd.read_csv('./toxic/labeled.csv')
toxic_comments.head()

toxic_comments.info()

TOKEN_RE = re.compile(r'[а-яё]+')
russian_stopwords = stopwords.words("russian")
lemmatizer = pymorphy2.MorphAnalyzer()

def tokenize_text(txt, min_lenght_token=2):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_lenght_token]

def remove_stopwords(tokens):
    return list(filter(lambda token: token not in russian_stopwords, tokens))

def lemmatizing(tokens):
    return [lemmatizer.parse(token)[0].normal_form for token in tokens]

def text_cleaning(txt):
    tokens = tokenize_text(txt)
    tokens  = lemmatizing(tokens)
    tokens = remove_stopwords(tokens)
    return ' '.join(tokens)

tqdm.pandas()

df_token = toxic_comments.copy()
df_token['comment'] = df_token['comment'].progress_apply(text_cleaning)
df_token.head()

df = df_token.copy()
empty = df[df['comment'] == '']
df = df.drop(empty.index)
df = df.drop_duplicates()

comment_duplicated = df[df['comment'].duplicated('last')]

# remove duplicate comments 
df = df.drop_duplicates(subset='comment')

# Labeling examples
zero_labels = [1084, 1198, 1250, 1394, 1456, 1586, 1631, 1637, 1659, 
               1693, 1703, 1739, 1781, 1814, 1820, 1877, 4256, 6044]
for row in comment_duplicated.iterrows():
    comment = row[1]['comment']
    idx = df[df['comment'] == comment].index
    if idx in zero_labels:
        label = 0.
    else:
        label = 1.
    df.loc[idx, 'toxic'] = label
    
print('Number of duplicates:', df.duplicated('comment').sum())

clean_text = df.copy()

corpus = clean_text['comment'].values

text = ' '.join(corpus)
counter = Counter(text.split())
sorted_counter = counter.most_common()


only_toxic = clean_text[clean_text['toxic'] == 1]
text_toxic = ' '.join(only_toxic['comment'].values)

df_train, df_test = train_test_split(clean_text, 
                                     random_state=311, 
                                     test_size=0.33, 
                                     stratify=clean_text['toxic']
                                    )

train_corpus = df_train['comment'].values
test_corpus = df_test['comment'].values

y_train = df_train['toxic']
y_test = df_test['toxic']

vectorizer = TfidfVectorizer(ngram_range=(2,4), analyzer='char_wb', max_df=0.8, min_df=10)
X_train = vectorizer.fit_transform(train_corpus)
X_test = vectorizer.transform(test_corpus)

print('Total features: ', len(vectorizer.get_feature_names()))

Logregres = LogisticRegression(max_iter=10000, C=3, solver='liblinear')
Logregres.fit(X_train, y_train)
y_pred = Logregres.predict(X_test)
print(classification_report(y_test, y_pred))

# отправляем в переменную всё содержимое текстового файла
text = open('./toxic/priped.txt', encoding='utf8').read()

# разбиваем текст на отдельные слова (знаки препинания останутся рядом со своими словами)
corpus = text.split()

# делаем новую функцию-генератор, которая определит пары слов
def make_pairs(corpus):
    # перебираем все слова в корпусе, кроме последнего
    for i in range(len(corpus)-1):
        # генерируем новую пару и возвращаем её как результат работы функции
        yield (corpus[i], corpus[i+1])
        
def make_comment():
    # вызываем генератор и получаем все пары слов
    pairs = make_pairs(corpus)

    # словарь, на старте пока пустой
    word_dict = dict(enumerate(only_toxic['comment'].values.flatten(), 10))
    #word_dict = {}

    # перебираем все слова попарно из нашего списка пар
    for word_1, word_2 in pairs:
        # если первое слово уже есть в словаре
        if word_1 in word_dict.keys():
            # то добавляем второе слово как возможное продолжение первого
            word_dict[word_1].append(word_2)
        # если же первого слова у нас в словаре не было
        else:
            # создаём новую запись в словаре и указываем второе слово как продолжение первого
            word_dict[word_1] = [word_2]

    # случайно выбираем первое слово для старта
    first_word = np.random.choice(corpus)

    # если в нашем первом слове нет больших букв 
    while first_word.islower():
        # то выбираем новое слово случайным образом
        # и так до тех пор, пока не найдём слово с большой буквой
        first_word = np.random.choice(corpus)

    # делаем наше первое слово первым звеном
    chain = [first_word]

    # сколько слов будет в готовом тексте
    n_words = 35

    # делаем цикл с нашим количеством слов
    for i in range(n_words):
        # на каждом шаге добавляем следующее слово из словаря, выбирая его случайным образом из доступных вариантов
        chain.append(np.random.choice(word_dict[chain[-1]]))

    # выводим результат
    message = ' '.join(chain)

    return message

message = make_comment()
clean_message = text_cleaning(message)
X_example = vectorizer.transform([clean_message])
toxic_propabality = Logregres.predict_proba(X_example)[0,1]

def getSentenceCase(source: str):
    output = ""
    isFirstWord = True

    for c in source:
        if isFirstWord and not c.isspace():
            c = c.upper()
            isFirstWord = False
        elif not isFirstWord and c in ".!?":
            isFirstWord = True
        else:
            c = c.lower()

        output = output + c

    return output

i = 1

while i < 5:
    message = '\n '+make_comment()
    message = getSentenceCase(message)
    clean_message = text_cleaning(message)
    X_example = vectorizer.transform([clean_message])
    toxic_propabality = Logregres.predict_proba(X_example)[0,1]
    #url = "https://api.telegram.org/bot5422061989:AAHp2EiYmibgMDoT-ECEs-8CdvFAL6Rig6c/sendMessage?chat_id=-1001689427652&text="+message
    #test = requests.post(url)
    print(message)
    print(f'Токсичность: {toxic_propabality:.2f}')
    i += 1