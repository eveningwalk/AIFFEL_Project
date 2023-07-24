----------------------------------------------

- 코더 : 남희정
- 리뷰어 : 김동규

----------------------------------------------

PRT(PeerReviewTemplate)

- [x] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

| 평가문항                                            | 상세기준                                                                    |
|-------------------------------------------------|------------------------------------------------------------------------|
| 1. 한글 코퍼스를 가공하여 BERT pretrain용 데이터셋을 잘 생성하였다. | MLM, NSP task의 특징이 잘 반영된 pretrain용 데이터셋 생성과정이 체계적으로 진행되었다. |
| 2. 구현한 BERT 모델의 학습이 안정적으로 진행됨을 확인하였다.       | 학습진행 과정 중에 MLM, NSP loss의 안정적인 감소가 확인되었다.                 |
| 3. 1M짜리 mini BERT 모델의 제작과 학습이 정상적으로 진행되었다.  | 학습된 모델 및 학습과정의 시각화 내역이 제출되었다.                        |


1번 항목  
최종 처리 결과를 확인한 결과 전처리가 된 것으로 보임

```python
{'tokens': ['[CLS]', '[MASK]', '▁~', '[MASK]', '[MASK]', '▁민주', '당', '▁출신', '[MASK]', '▁3', '9', '번째', '▁대통령', '▁(19', '7', '7', '년', '▁~', '▁1981', '년', ')', '이다', '.', '[MASK]', '[MASK]', '▁카', '터', '는', '▁조지', '아', '주', '▁섬', '터', '▁카운', '티', '▁플', '레', '인', '스', '[MASK]', '[MASK]', '▁태어났다', '.', '[MASK]', '[MASK]', '▁공', '과', '대학교', '를', '▁졸업', '하였다', '.', '[MASK]', '▁후', '[MASK]', '[MASK]', '▁들어가', '▁전', '함', '·', '원', '자', '력', '·', '잠', '수', '함', '의', '▁승', '무', '원으로', '▁일', '하였다', '.', '▁195', '3', '년', '[MASK]', '▁해군', '▁대', '위로', '▁예', '편', '하였고', '▁이후', '▁땅', '콩', '·', '면', '화', '[MASK]', '▁가', '꿔', '▁많은', '▁돈', '을', '▁벌', '었다', '.', '▁그의', '▁별', '명이', '▁"', '땅', '콩', '▁농', '부', '"', '▁(', 'P', 'e', 'an', 'ut', '▁F', 'ar', 'm', 'er', ')', '로', '▁알려', '졌다', '.', '[SEP]', '▁지', '미', '▁카', '터', '[SEP]'], 'segment': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'is_next': 0, 'mask_idx': [1, 3, 4, 8, 23, 24, 39, 40, 43, 44, 52, 54, 55, 77, 90, 119, 120, 121], 'mask_label': ['일', '▁)', '는', '▁미국', '▁지', '미', '▁마을', '에서', '▁조지', '아', '▁그', '▁해', '군에', '▁미국', '▁등을', '▁알려', '졌다', '.']}
enc_token: [5, 6, 203, 6, 6, 1114, 3724, 788, 6, 49, 3632, 796, 663, 1647, 3682, 3682, 3625, 203, 3008, 3625, 3616, 16, 3599, 6, 6, 207, 3714, 3602, 1755, 3630, 3646, 630, 3714, 3565, 3835, 429, 3740, 3628, 3626, 6, 6, 1605, 3599, 6, 6, 41, 3644, 830, 3624, 1135, 52, 3599, 6, 81, 6, 6, 2247, 25, 3779, 3873, 3667, 3631, 3813, 3873, 4196, 3636, 3779, 3601, 249, 3725, 1232, 33, 52, 3599, 479, 3652, 3625, 6, 2780, 14, 1509, 168, 3877, 414, 165, 1697, 4290, 3873, 3703, 3683, 6, 21, 5007, 399, 1927, 3607, 813, 17, 3599, 307, 587, 931, 103, 4313, 4290, 613, 3638, 3718, 98, 3878, 3656, 256, 2543, 309, 337, 3735, 181, 3616, 3603, 489, 376, 3599, 4, 18, 3686, 207, 3714, 4]
segment: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
label_nsp: 0
label_mlm: [   0 3629    0  241 3602    0    0    0  243    0    0    0    0    0
    0    0    0    0    0    0    0    0    0   18 3686    0    0    0
    0    0    0    0    0    0    0    0    0    0    0 1369   10    0
    0 1755 3630    0    0    0    0    0    0    0   13    0   87 1501
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0  243    0    0    0    0    0    0
    0    0    0    0    0    0  593    0    0    0    0    0    0    0
    0    0    0    0    0    0    0    0    0    0    0    0    0    0
    0    0    0    0    0    0    0  489  376 3599    0    0    0    0
    0    0]
```


2번 항목  
로그를 확인한 결과 충분히 완만하게 감소한 것으로 보인다.
```python
Epoch 7/10
2000/2000 [==============================] - 251s 126ms/step - loss: 12.5703 - nsp_loss: 0.5965 - mlm_loss: 11.9738 - nsp_acc: 0.6586 - mlm_lm_acc: 0.2393

Epoch 8/10
2000/2000 [==============================] - 251s 126ms/step - loss: 12.3625 - nsp_loss: 0.5912 - mlm_loss: 11.7713 - nsp_acc: 0.6681 - mlm_lm_acc: 0.2467

Epoch 9/10
2000/2000 [==============================] - 251s 126ms/step - loss: 12.2338 - nsp_loss: 0.5863 - mlm_loss: 11.6475 - nsp_acc: 0.6766 - mlm_lm_acc: 0.2507

Epoch 10/10
2000/2000 [==============================] - 251s 126ms/step - loss: 12.1764 - nsp_loss: 0.5839 - mlm_loss: 11.5924 - nsp_acc: 0.6817 - mlm_lm_acc: 0.2525
```

3번 항목  
시각화 코드가 제공 되었고 그 결과를 확인 할 수 있었다.

![download](https://github.com/crlotwhite-mirror/AIFFEL_Project/assets/133851227/0f85ecf5-98cf-44fb-bfab-f2cbbe752e54)

- [x] 주석을 보고 작성자의 코드가 이해되었나요?

작은 코드들에도 이렇게 docstr을 달아서 어떤 역할을 수행하는지 파악 가능함

```python
@tf.function(experimental_relax_shapes=True)
def gelu(x):
    """
    gelu activation 함수
    :param x: 입력 값
    :return: gelu activation result
    """
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def kernel_initializer(stddev=0.02):
    """
    parameter initializer 생성
    :param stddev: 생성할 랜덤 변수의 표준편차
    """
    return tf.keras.initializers.TruncatedNormal(stddev=stddev)


def bias_initializer():
    """
    bias initializer 생성
    """
    return tf.zeros_initializer
```

- [x] 코드가 에러를 유발할 가능성이 있나요?

env를 통해, home 폴더를 지정하므로서 절대 경로 사용시 발생가능한 에러를 미연에 방지함

```
corpus_file = os.getenv('HOME')+'/aiffel/bert_pretrain/data/kowiki.txt'
```

- [x] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)

주로 사용하는 학습 코드를 한 셀에 넣어서 실험마다 필요한 작업을 일괄 실행하도록함.  
이것이 가능하다는 것은 코드를 이해하고 있다고 볼 수 있음.

```
# compute lr 
test_schedule = CosineSchedule(train_steps=4000, warmup_steps=500)
lrs = []
for step_num in range(4000):
    lrs.append(test_schedule(float(step_num)).numpy())

# draw
plt.plot(lrs, 'r-', label='learning_rate')
plt.xlabel('Step')
plt.show()
     

# 모델 생성
pre_train_model = build_model_pre_train(config)
pre_train_model.summary()
     

epochs = 10
batch_size = 64

# optimizer
train_steps = math.ceil(len(pre_train_inputs[0]) / batch_size) * epochs
print("train_steps:", train_steps)
learning_rate = CosineSchedule(train_steps=train_steps, warmup_steps=max(100, train_steps // 10))
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

# compile
pre_train_model.compile(loss=(tf.keras.losses.sparse_categorical_crossentropy, lm_loss), optimizer=optimizer, metrics={"nsp": "acc", "mlm": lm_acc})
     

# save weights callback
save_weights = tf.keras.callbacks.ModelCheckpoint(f"{model_dir}/bert_pre_train.hdf5", monitor="mlm_lm_acc", verbose=1, save_best_only=True, mode="max", save_freq="epoch", save_weights_only=True)
# train
history = pre_train_model.fit(pre_train_inputs, pre_train_labels, epochs=epochs, batch_size=batch_size, callbacks=[save_weights])
     

# training result
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['nsp_loss'], 'b-', label='nsp_loss')
plt.plot(history.history['mlm_loss'], 'r--', label='mlm_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['nsp_acc'], 'g-', label='nsp_acc')
plt.plot(history.history['mlm_lm_acc'], 'k--', label='mlm_acc')
plt.xlabel('Epoch')
plt.legend()

plt.show()
```

- [x] 코드가 간결한가요?

for 문을 사용해서 같은 코드를 사용하지 않았다.  
이를 통해 좀 더 간결하고 명확한 코드가 작성되었다.

```
enc_out = self.dropout(enc_embed)
for encoder_layer in self.encoder_layers:
    enc_out = encoder_layer(enc_out, enc_self_mask)

```
 
 ----------------------------------------------

참고 링크 및 코드 개선
