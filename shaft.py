import pandas as pd

val   = pd.read_csv('./shaft/val.csv')
train = pd.read_csv('./shaft/train.csv')
test  = pd.read_csv('./shaft/test.csv')

df = pd.concat([val, train, test])
'lines %d, columns %d' % df.shape

df = df.drop_duplicates(subset=['text'])
'lines %d, columns %d' % df.shape

df[['text', 'inappropriate']].sample(3)

a = df['inappropriate'] > 0.8
b = df['gambling'] > 0.8

sample = df[a & b].sample(1)
sample['text']

df.columns

columns = ['suicide', 'body_shaming', 'health_shaming', 'politics', 'religion', 'racism', 'social_injustice', 'online_crime']
df = df.drop(columns, axis=1)
df.columns

df.sample(3)

unwanted = ['human_labeled', 'text', 'inappropriate']
shaft_columns = [i for i in df.columns.tolist() if not i in unwanted]
shaft_columns

sample = df.iloc[1]
sample


def shaft_label(content, shaft_columns):
    a = set(content.index)
    b = set(shaft_columns)
    columns = a & b
    
    if any([content[c] > 0.75 for c in columns]) and content['inappropriate'] > 0.75:
        return 'pos'
    elif any([content[c] > 0.4 for c in columns]) and content['inappropriate'] > 0.4:
        return 'unk'
    return 'neg'

shaft_label(sample, shaft_columns)

df['shaft'] = df.apply(lambda x: shaft_label(x, shaft_columns), axis=1)
df = df[['text', 'shaft', 'human_labeled']].copy()
df.sample(3)

df['shaft'].value_counts()

df = df[df['shaft'] != 'unk'].copy()
'lines %d, columns %d' % df.shape

N = 5
positive = df[df['shaft']=='pos']
texts = positive['text'].sample(N).tolist()
for i, t in enumerate(texts):
    print(i+1, t)

df['human_labeled'].value_counts()
#df.to_csv('shaft-russian.csv', index=False)