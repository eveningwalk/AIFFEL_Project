>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|SentencePiece를 이용하여 모델을 만들기까지의 과정이 정상적으로 진행되었는가?|코퍼스 분석, 전처리, SentencePiece 적용, 토크나이저 구현 및 동작이 빠짐없이 진행되었는가?|⭐|
>|2|SentencePiece를 통해 만든 Tokenizer가 자연어처리 모델과 결합하여 동작하는가?|SentencePiece 토크나이저가 적용된 Text Classifier 모델이 정상적으로 수렴하여 80% 이상의 test accuracy가 확인되었다.||
>|3|SentencePiece의 성능을 다각도로 비교분석하였는가?|SentencePiece 토크나이저를 활용했을 때의 성능을 다른 토크나이저 혹은 SentencePiece의 다른 옵션의 경우와 비교하여 분석을 체계적으로 진행하였다.||

----------------------------------------------

- 코더 : 남희정
- 리뷰어 : 김경훈

----------------------------------------------

PRT(PeerReviewTemplate)

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

* Mecab을 사용한 전처리에서 vocab_size 문제
``` python
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
m = Mecab()

def get_mecab_tokenize(data):
    mecab_corpus = []
    
    for sentence in data['document']:
        mecab_corpus.append(m.morphs(sentence))

    tokenizer.fit_on_texts(mecab_corpus)
    mecab_tensor = tokenizer.texts_to_sequences(mecab_corpus)
    mecab_tensor = tf.keras.preprocessing.sequence.pad_sequences(mecab_tensor, padding='post', maxlen=max_len)
    return mecab_tensor

x_train = get_mecab_tokenize(filtered_corpus)
```
> 위 코드에서 `x_train`의 vocab_size가 10,000이 아닐텐데 get_model() 함수에서 그대로 사용하고 있습니다. 

- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [ ] 코드가 간결한가요? 
 
 ----------------------------------------------

참고 링크 및 코드 개선

