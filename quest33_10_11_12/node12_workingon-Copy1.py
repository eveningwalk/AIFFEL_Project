#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import tensorflow
import matplotlib

print(pd.__version__)
print(tensorflow.__version__)
print(matplotlib.__version__)


# In[2]:


import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

import time
import re
import os
import io

print(tf.__version__)


# In[3]:


path_en = './data/korean-english-park.train.en'
path_kor = './data/korean-english-park.train.ko'

raw_kor = pd.read_table(path_kor, header= None)
raw_en = pd.read_table(path_en, header = None)

train_data = pd.concat([raw_kor, raw_en], axis=1)


# In[4]:


column_labels = ['Korean', 'English']
train_data.columns = column_labels

train_data.head(30)


# ### 중복 제외

# In[5]:


print('Train dataset size :',len(train_data))

train_data.drop_duplicates(subset=['Korean'], inplace=True)
print('Train dataset size :',len(train_data))

train_data.drop_duplicates(subset=['English'], inplace=True)
print('Train dataset size :',len(train_data))

train_data.dropna(how = 'any')   
print('Train dataset size :',len(train_data))


# ### 문장 길이 40이하 선별

# In[6]:


def sentence_length_stats(data):
    mask = data['Korean'].str.len() <= 40
    result = data[mask]

    return result


# In[7]:


train_data = sentence_length_stats(train_data)
train_data.head(10)


# In[8]:


print('Train dataset size :',len(train_data))


# ### Preprocessing

# In[9]:


#For korean

import re

def preprocess_sentence_kor(sentence, s_token=False, e_token=False):
    sentence = sentence.strip()

    # Add spaces around punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # Remove consecutive spaces
    sentence = re.sub(r'[" "]+', " ", sentence)

    # Remove all characters except Korean, spaces, and punctuation
    sentence = re.sub(r"[^가-힣0-9a-zA-Z?.!,]+", " ", sentence)

    # Remove leading/trailing spaces
    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'

    return sentence


# In[10]:


#For English

def preprocess_sentence_en(sentence, s_token=False, e_token=False):
    sentence = sentence.lower().strip()

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)

    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'
    
    return sentence


# In[11]:


enc_corpus = []
dec_corpus = []

num_examples = 30000

for line in train_data['Korean']:
    enc_corpus.append(preprocess_sentence_kor(line))
    
for line in train_data['English']:
    dec_corpus.append(preprocess_sentence_en(str(line), s_token=True, e_token=True))


# In[12]:


print("Korean:", enc_corpus[1])   # go away !
print("English:", dec_corpus[1])   # <start> salga de aqu ! <end>


# In[13]:


len(enc_corpus),len(dec_corpus)


# ### Tokenizing

# In[14]:


from konlpy.tag import Mecab


# In[15]:


def tokenize_en(corpus):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(corpus)

    tensor = tokenizer.texts_to_sequences(corpus)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, tokenizer


# In[16]:


def tokenize_kor(corpus):
    
    max_len = 40
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    m = Mecab()

    mecab_corpus = []
    
    for sentence in corpus:
        mecab_corpus.append(m.morphs(sentence))

    tokenizer.fit_on_texts(mecab_corpus)
    mecab_tensor = tokenizer.texts_to_sequences(mecab_corpus)
    mecab_tensor = tf.keras.preprocessing.sequence.pad_sequences(mecab_tensor, padding='post', maxlen=max_len)
    return mecab_tensor, tokenizer


# In[17]:


# 토큰화하기
enc_tensor, enc_tokenizer = tokenize_kor(enc_corpus)
dec_tensor, dec_tokenizer = tokenize_en(dec_corpus)


# train_test_split을 활용해서 훈련 데이터와 검증 데이터로 분리하기

from sklearn.model_selection import train_test_split
enc_train, enc_val = train_test_split(enc_tensor, test_size=0.2)
dec_train, dec_val = train_test_split(dec_tensor, test_size=0.2)


# In[18]:


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.w_dec = tf.keras.layers.Dense(units)
        self.w_enc = tf.keras.layers.Dense(units)
        self.w_com = tf.keras.layers.Dense(1)
    
    def call(self, h_enc, h_dec):
        # h_enc shape: [batch x length x units]
        # h_dec shape: [batch x units]

        h_enc = self.w_enc(h_enc)
        h_dec = tf.expand_dims(h_dec, 1)
        h_dec = self.w_dec(h_dec)

        score = self.w_com(tf.nn.tanh(h_dec + h_enc))
        
        attn = tf.nn.softmax(score, axis=1)

        context_vec = attn * h_enc
        context_vec = tf.reduce_sum(context_vec, axis=1)

        return context_vec, attn


# In[19]:


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True)
        
    def call(self, x):
        x = self.embedding(x)
        x = self.gru(x)
        return x


# In[20]:


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units,
                                       return_sequences=True,
                                       return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, h_dec, enc_out):
        context_vec, attn = self.attention(enc_out, h_dec)

        out = self.embedding(x)
        context_vec = tf.expand_dims(context_vec, 1)
        out = tf.concat([context_vec, out], axis=-1)

        out, h_dec = self.gru(out)
        out = tf.reshape(out, (-1, out.shape[2]))
        out = self.fc(out)
        return out, h_dec, attn


# import tensorflow as tf
# 
# class Encoder(tf.keras.Model):
#   def __init__(self, vocab_size, embedding_dim, enc_units):
#     super(Encoder, self).__init__()
#     self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#     self.lstm = tf.keras.layers.LSTM(enc_units) # return_sequences 매개변수를 기본값 False로 전달
# 
#   def call(self, x):
#     print("입력 Shape:", x.shape)
# 
#     x = self.embedding(x)
#     print("Embedding Layer를 거친 Shape:", x.shape)
# 
#     output = self.lstm(x)
#     print("LSTM Layer의 Output Shape:", output.shape)
# 
#     return output
# 
# print("슝~")

# class Decoder(tf.keras.Model):
#     def __init__(self, vocab_size, embedding_dim, dec_units):
#         super(Decoder, self).__init__()
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
#         self.lstm = tf.keras.layers.LSTM(dec_units,
#                                          return_sequences=True) # return_sequences 매개변수를 True로 설정
#         self.fc = tf.keras.layers.Dense(vocab_size)
#         self.softmax = tf.keras.layers.Softmax(axis=-1)
# 
#     def call(self, x, h_dec, context_v):  # 디코더의 입력 x와 인코더의 컨텍스트 벡터를 인자로 받는다. 
#         print("입력 Shape:", x.shape)
# 
#         x = self.embedding(x)
#         print("Embedding Layer를 거친 Shape:", x.shape)
# 
#         context_v = tf.repeat(tf.expand_dims(context_v, axis=1),
#                               repeats=x.shape[1], axis=1)
#         x = tf.concat([x, context_v], axis=-1)  # 컨텍스트 벡터를 concat 해준다
#         print("Context Vector가 더해진 Shape:", x.shape)
# 
#         x = self.lstm(x)
#         print("LSTM Layer의 Output Shape:", x.shape)
# 
#         output = self.fc(x)
#         print("Decoder 최종 Output Shape:", output.shape)
# 
#         return self.softmax(output)
# 

# In[21]:


# 코드를 실행하세요.

BATCH_SIZE     = 64
SRC_VOCAB_SIZE = len(enc_tokenizer.index_word) + 1
TGT_VOCAB_SIZE = len(dec_tokenizer.index_word) + 1

units         = 1024
embedding_dim = 512

encoder = Encoder(SRC_VOCAB_SIZE, embedding_dim, units )
decoder = Decoder(TGT_VOCAB_SIZE, embedding_dim, units)

# sample input
sequence_len = 30

sample_enc = tf.random.uniform((BATCH_SIZE, sequence_len))
sample_output = encoder(sample_enc)

print ('Encoder Output:', sample_output.shape)

sample_state = tf.random.uniform((BATCH_SIZE, units))

sample_logits, h_dec, attn = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                     sample_state, sample_output)

print ('Decoder Output:', sample_logits.shape)
print ('Decoder Hidden State:', h_dec.shape)
print ('Attention:', attn.shape)


# In[22]:


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_mean(loss)


# In[23]:


@tf.function
def train_step(src, tgt, encoder, decoder, optimizer, dec_tok):
    bsz = src.shape[0]
    loss = 0

    with tf.GradientTape() as tape:
        enc_out = encoder(src)
        h_dec = enc_out[:, -1]
        
        dec_src = tf.expand_dims([dec_tok.word_index['<start>']] * bsz, 1)

        for t in range(1, tgt.shape[1]):
            pred, h_dec, _ = decoder(dec_src, h_dec, enc_out)

            loss += loss_function(tgt[:, t], pred)
            dec_src = tf.expand_dims(tgt[:, t], 1)
        
    batch_loss = (loss / int(tgt.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return batch_loss


# from tqdm import tqdm    # tqdm
# import random
# 
# EPOCHS = 1
# 
# for epoch in range(EPOCHS):
#     total_loss = 0
#     
#     idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
#     random.shuffle(idx_list)
#     t = tqdm(idx_list)    # tqdm
# 
#     for (batch, idx) in enumerate(t):
#         batch_loss = train_step(enc_train[idx:idx+BATCH_SIZE],
#                                 dec_train[idx:idx+BATCH_SIZE],
#                                 encoder,
#                                 decoder,
#                                 optimizer,
#                                 dec_tokenizer)
#     
#         total_loss += batch_loss
#         
#         t.set_description_str('Epoch %2d' % (epoch + 1))    # tqdm
#         t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))    # tqdm

# In[ ]:


# eval_step() 정의하기
@tf.function
def eval_step(src, tgt, encoder, decoder, dec_tok):
    bsz = src.shape[0]
    loss = 0

    enc_out = encoder(src)

    h_dec = enc_out[:, -1]
    
    #dec_src = tf.expand_dims([dec_tok.word_index['']] * bsz, 1)
    dec_src = tf.expand_dims([dec_tok.word_index['<start>']] * bsz, 1)

    #print(tgt.shape[1])
    for t in range(1, tgt.shape[1]):
        pred, h_dec, _ = decoder(dec_src, h_dec, enc_out)

        loss += loss_function(tgt[:, t], pred)
        dec_src = tf.expand_dims(tgt[:, t], 1)
        
    batch_loss = (loss / int(tgt.shape[1]))
    
    return batch_loss


# train_step() 이후 eval_step() 진행하도록 소스 수정하기
from tqdm import tqdm    # tqdm
import random

EPOCHS =10


for epoch in range(EPOCHS):
    total_loss = 0
    
    idx_list = list(range(0, enc_train.shape[0], BATCH_SIZE))
    random.shuffle(idx_list)
    t = tqdm(idx_list)

    for (batch, idx) in enumerate(t):
        batch_loss = train_step(enc_train[idx:idx+BATCH_SIZE],
                                dec_train[idx:idx+BATCH_SIZE],
                                encoder,
                                decoder,
                                optimizer,
                                dec_tokenizer)
    
        total_loss += batch_loss
        
        t.set_description_str('Epoch %2d' % (epoch + 1))
        t.set_postfix_str('Loss %.4f' % (total_loss.numpy() / (batch + 1)))
    
    test_loss = 0
    
    idx_list = list(range(0, enc_val.shape[0], BATCH_SIZE))
    random.shuffle(idx_list)
    t = tqdm(idx_list)
    for (test_batch, idx) in enumerate(t):
        test_batch_loss = eval_step(enc_val[idx:idx+BATCH_SIZE],
                                    dec_val[idx:idx+BATCH_SIZE],
                                    encoder,
                                    decoder,
                                    dec_tokenizer)
    
        test_loss += test_batch_loss

        t.set_description_str('Test Epoch %2d' % (epoch + 1))
        t.set_postfix_str('Test Loss %.4f' % (test_loss.numpy() / (test_batch + 1)))


# In[ ]:


def evaluate(sentence, encoder, decoder):
    attention = np.zeros((dec_train.shape[-1], enc_train.shape[-1]))
    
    sentence = preprocess_sentence(sentence)
    inputs = enc_tokenizer.texts_to_sequences([sentence.split()])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                           maxlen=enc_train.shape[-1],
                                                           padding='post')

    result = ''

    enc_out = encoder(inputs)

    dec_hidden = enc_out[:, -1]
    dec_input = tf.expand_dims([dec_tokenizer.word_index['<start>']], 0)

    for t in range(dec_train.shape[-1]):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention[t] = attention_weights.numpy()

        predicted_id =         tf.argmax(tf.math.softmax(predictions, axis=-1)[0]).numpy()

        result += dec_tokenizer.index_word[predicted_id] + ' '

        if dec_tokenizer.index_word[predicted_id] == '<end>':
            return result, sentence, attention

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention


def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}
    
    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(sentence) + 1)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(predicted_sentence) + 1)))
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def translate(sentence, encoder, decoder):
    result, sentence, attention = evaluate(sentence, encoder, decoder)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))
    
    attention = attention[:len(result.split()), :len(sentence.split())]
    plot_attention(attention, sentence.split(), result.split(' '))


translate("돌리는데 시간이 너무 오래 걸리네요", encoder, decoder)

