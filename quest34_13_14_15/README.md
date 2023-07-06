----------------------------------------------

- 코더 : 남희정
- 리뷰어 : 김동규
- visuzlization 파일은 그레프때문에 코드가 너무 길어져서 중간에 interrupt 한 상태의 결과입니다. 참고바랍니다. 
----------------------------------------------

## PRT(PeerReviewTemplate)

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [x] 코드가 간결한가요? 


## 판단 근거

### 항목 1
#### 루브릭
| 평가문항 | 상세기준 |
|---|---|
| 번역기 모델 학습에 필요한 텍스트 데이터 전처리가 잘 이루어졌다. | 데이터 정제, SentencePiece를 활용한 토큰화 및 데이터셋 구축의 과정이 지시대로 진행되었다. |
| Transformer 번역기 모델이 정상적으로 구동된다. | Transformer 모델의 학습과 추론 과정이 정상적으로 진행되어, 한-영 번역기능이 정상 동작한다. |
| 테스트 결과 의미가 통하는 수준의 번역문이 생성되었다. | 제시된 문장에 대한 그럴듯한 영어 번역문이 생성되며, 시각화된 Attention Map으로 결과를 뒷받침한다. |

#### 기준 1

**중복 제거**
```python
#병렬데이터 중복 제거
def duplication_remover(enc_corpus, dec_corpus ):
    cleaned_corpus = set(zip(enc_corpus, dec_corpus))
    see = list(cleaned_corpus)
    see[:5]
    return list(cleaned_corpus)
```

**정규식을 이용한 전처리**
```python
def preprocess_sentence_kor(sentence): 
    sentence = sentence.strip()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^가-힣0-9a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()

    return sentence
```
영어 버전도 따로 있으나 첨부는 생략함

**Sentencepiece 활용한 토크나이징**
```python
# Sentencepiece를 활용하여 학습한 tokenizer를 생성합니다.
def generate_tokenizer(corpus, vocab_size, lang="ko", pad_id=0, bos_id=1, eos_id=2, unk_id=3):
    sentencepiece_learning(corpus, vocab_size, lang, pad_id, bos_id, eos_id, unk_id)

    word_index, index_word = read_sp_vocab(lang)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(f"{lang}.model")
    
    tokenizer = sp_tokenize(tokenizer, corpus)
    
    return tokenizer
```
세부 코드는 모두 정상적으로 작성되었으며, 글이 쓸대없이 길어지므로 첨부를 생략한다.

**정수화 및 패딩 시퀀스**
```python
src_corpus = []
tgt_corpus = []

assert len(kor_corpus) == len(eng_corpus)

# Keep only sentences with token length less than or equal to 50
for idx in tqdm(range(len(kor_corpus))):
    kor_tokenized = ko_tokenizer.EncodeAsIds(kor_corpus[idx])
    eng_tokenized = en_tokenizer.EncodeAsIds(eng_corpus[idx])
    if len(kor_tokenized) <= 50 and len(eng_tokenized) <= 50:
        src_corpus.append(kor_tokenized)
        tgt_corpus.append(eng_tokenized)

# Pad the sequences to create training data
enc_train = tf.keras.preprocessing.sequence.pad_sequences(src_corpus, padding='post')
dec_train = tf.keras.preprocessing.sequence.pad_sequences(tgt_corpus, padding='post')
```
모두 정상적으로 작성된 것으로 보인다.

이러한 근거를 통해 기준 1을 달성했다고 볼 수 있다.

#### 기준 2

**모델 설계**
```python
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)
        
    def call(self, x, mask):

        """
        Multi-Head Attention
        """
        residual = x
        out = self.norm_1(x)
        out, enc_attn = self.enc_self_attn(out, out, out, mask)
        out = self.dropout(out)
        out += residual
        
        """
        Position-Wise Feed Forward Network
        """
        residual = out
        out = self.norm_2(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual
        
        return out, enc_attn

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)

        self.ffn = PoswiseFeedForwardNet(d_model, d_ff)

        self.norm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, x, enc_out, causality_mask, padding_mask):

        """
        Masked Multi-Head Attention
        """
        residual = x
        out = self.norm_1(x)
        out, dec_attn = self.dec_self_attn(out, out, out, padding_mask)
        out = self.dropout(out)
        out += residual

        """
        Multi-Head Attention
        """
        residual = out
        out = self.norm_2(out)
        out, dec_enc_attn = self.enc_dec_attn(out, enc_out, enc_out, causality_mask)
        out = self.dropout(out)
        out += residual
        
        """
        Position-Wise Feed Forward Network
        """
        residual = out
        out = self.norm_3(out)
        out = self.ffn(out)
        out = self.dropout(out)
        out += residual

        return out, dec_attn, dec_enc_attn
```
전반적으로 잘 구성된 것으로 보인다.  
첨부 자료로서 가장 핵심구조라고 생각될 수 있는 '인코더'와 '디코더'를 첨부한다.  
첨부되지 않은 코드에 대해서 문제가 있는 것은 아니다.

아래의 코드는 옵티마이저 구현이고 잘 쓰여 있다.
```
class LearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(LearningRateScheduler, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        arg1 = step ** -0.5
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return (self.d_model ** -0.5) * tf.math.minimum(arg1, arg2)

learning_rate = LearningRateScheduler(512)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98, 
                                     epsilon=1e-9)
```

**번역 결과**
```
Input: 오바마는 대통령이다.
Predicted translation: <start> obama . <end>
Input: 시민들은 도시 속에 산다.
Predicted translation: <start> the city s visit is a city on the san each san each . <end>
Input: 커피는 필요 없다.
Predicted translation: <start> no coffee is needed . <end>
Input: 일곱 명의 사망자가 발생했다.
Predicted translation: <start> seven people died . <end
```

번역 결과가 충분히 acceptable한 결과를 보여준다.  
특히 3번과 4번 문항은 거의 정확하게 번역됬다고 생각한다.

#### 기준 3
README에 쓰인 것 처럼 중간에 스톱한 노트라 그래프가 없다고 한다.  
번역 결과를 미루어보아 어텐션 그래프는 충분히 잘 나올 것으로 보인다.

### 항목 2
각 스텝에 대한 주석이 있어서 이해가 쉽다.
```python
def preprocess_sentence_kor(sentence): 
    sentence = sentence.strip()

    # Add spaces around punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # Remove consecutive spaces
    sentence = re.sub(r'[" "]+', " ", sentence)

    # Remove all characters except Korean, spaces, and punctuation
    sentence = re.sub(r"[^가-힣0-9a-zA-Z?.!,]+", " ", sentence)

    
    # Remove leading/trailing spaces
    sentence = sentence.strip()

    #if s_token:
    #    sentence = '<start> ' + sentence

    #if e_token:
    #    sentence += ' <end>'

    return sentence
```

### 항목 3
inner-function을 사용해서 함수 이용의 스코프를 명확하게 제한했다.  
다른 코드에서 오용될 확률이 감소했다.

```python
def visualize_attention(src, tgt, enc_attns, dec_attns, dec_enc_attns):
    def draw(data, ax, x="auto", y="auto"):
        import seaborn
        seaborn.heatmap(data, 
                        square=True,
                        vmin=0.0, vmax=1.0, 
                        cbar=False, ax=ax,
                        xticklabels=x,
                        yticklabels=y)
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Encoder Layer", layer + 1)
        for h in range(4):
            draw(enc_attns[layer][0, h, :len(src), :len(src)], axs[h], src, src)
        plt.show()
        
    for layer in range(0, 2, 1):
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        print("Decoder Self Layer", layer+1)
        for h in range(4):
            draw(dec_attns[layer][0, h, :len(tgt), :len(tgt)], axs[h], tgt, tgt)
        plt.show()

        print("Decoder Src Layer", layer+1)
        fig, axs = plt.subplots(1, 4, figsize=(20, 10))
        for h in range(4):
            draw(dec_enc_attns[layer][0, h, :len(tgt), :len(src)], axs[h], src, tgt)
        plt.show()
``

### 항목 4

**상황에 맞는 정규식 이용함**
```python
#For English

def preprocess_sentence_en(sentence, s_token=False, e_token=False):
    sentence = sentence.lower().strip()

    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)

    # Replace to lower characters
    sentence = sentence.lower()
    
    sentence = sentence.strip()

    if s_token:
        sentence = '<start> ' + sentence

    if e_token:
        sentence += ' <end>'
    
    return sentence

def preprocess_sentence_kor(sentence): 
    sentence = sentence.strip()

    # Add spaces around punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # Remove consecutive spaces
    sentence = re.sub(r'[" "]+', " ", sentence)

    # Remove all characters except Korean, spaces, and punctuation
    sentence = re.sub(r"[^가-힣0-9a-zA-Z?.!,]+", " ", sentence)

    
    # Remove leading/trailing spaces
    sentence = sentence.strip()

    #if s_token:
    #    sentence = '<start> ' + sentence

    #if e_token:
    #    sentence += ' <end>'

    return sentence
```

### 항목 5
**for-statement를 사용하여 중복코드를 최소화함**
```python
src_corpus = []
tgt_corpus = []

assert len(kor_corpus) == len(eng_corpus)

# Keep only sentences with token length less than or equal to 50
for idx in tqdm(range(len(kor_corpus))):
    kor_tokenized = ko_tokenizer.EncodeAsIds(kor_corpus[idx])
    eng_tokenized = en_tokenizer.EncodeAsIds(eng_corpus[idx])
    if len(kor_tokenized) <= 50 and len(eng_tokenized) <= 50:
        src_corpus.append(kor_tokenized)
        tgt_corpus.append(eng_tokenized)

# Pad the sequences to create training data
enc_train = tf.keras.preprocessing.sequence.pad_sequences(src_corpus, padding='post')
dec_train = tf.keras.preprocessing.sequence.pad_sequences(tgt_corpus, padding='post')
```



 ----------------------------------------------

참고 링크 및 코드 개선
