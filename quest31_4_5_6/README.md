>## **루브릭**
>
>|번호|평가문항|상세기준|평가결과|
>|:---:|---|---|:---:|
>|1|분류 모델의 accuracy가 기준 이상 높게 나왔는가?|3가지 단어 개수에 대해 8가지 머신러닝 기법을 적용하여 그중 최적의 솔루션을 도출하였다.||
>|2|분류 모델의 F1 score가 기준 이상 높게 나왔는가?|Vocabulary size에 따른 각 머신러닝 모델의 성능변화 추이를 살피고, 해당 머신러닝 알고리즘의 특성에 근거해 원인을 분석하였다.||
>|3|딥러닝 모델을 활용해 성능이 비교 및 확인되었는가?|동일한 데이터셋과 전처리 조건으로 딥러닝 모델의 성능과 비교하여 결과에 따른 원인을 분석하였다.||


----------------------------------------------

- 코더 : 남희정
- 리뷰어 : 김경훈

----------------------------------------------

PRT(PeerReviewTemplate)

- [ ] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?

* 문법에러
``` python
# 원본
decoded = []
for i in range(len(x_train)):
    t = ' '.join([index_to_word[i] for index in x_train[i]])
    decoded.append(t)

# 수정
decoded = []
for i in range(len(x_train)):
    t = ' '.join([index_to_word[index] for index in x_train[i]])
    decoded.append(t)
```

* 결과를 같은 변수에 계속 업데이트 합니다. (변수를 수정해서 리턴시키던지, 프린트 하는 함수를 추가해야될 것 같습니다.)
``` python
def fit_ml(x_train, y_train, x_test, y_test):
    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    predicted = nb.predict(x_test) #테스트 데이터에 대한 예측
    
    cb = ComplementNB()
    cb.fit(x_train, y_train)
    predicted = cb.predict(x_test)
    
    lr = LogisticRegression(C=10000, penalty='l2', max_iter=3000)
    lr.fit(x_train, y_train)
    predicted = lr.predict(x_test)
    
    lsvc = LinearSVC(C=1000, penalty='l1', max_iter=3000, dual=False)
    lsvc.fit(x_train, y_train)
    predicted = lsvc.predict(x_test)
    
    tree = DecisionTreeClassifier(max_depth=10, random_state=0)
    tree.fit(x_train, y_train)
    predicted = tree.predict(x_test)
    
    forest = RandomForestClassifier(n_estimators=5, random_state=0)
    forest.fit(x_train, y_train)
    predicted = forest.predict(x_test)
    
    grbt = GradientBoostingClassifier(random_state=0) # verbose=3
    grbt.fit(x_train, y_train)
    predicted = grbt.predict(x_test)
    
    clf1 = LogisticRegression()
    clf2 = ComplementNB()
    clf3 = GradientBoostingClassifier(random_state=0)

    voting_classifier = VotingClassifier(estimators=[('lr', clf1), ('gnb', clf2), ('dt', clf3)], voting='hard')
    voting_classifier.fit(x_train, y_train)
    predicted = voting_classifier.predict(x_test)
```
- [x] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?

* `decode`를 같은 변수로 계속 사용하면 문제가 생길 수 있습니다.
* `dtmvector`, `tfidf_transformer` 를 전역변수로 사용하면 다른 num_words로 실행시 문제가 생길 수 있습니다.

원래 코드
``` python
dtmvector = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def reuters_load_ml(num_words):
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.2)
    
    decoded = []
    for i in range(len(x_train)):
        t = ' '.join([index_to_word[i] for index in x_train[i]])
        decoded.append(t)

    x_train = decoded
    
    decoded = []
    for i in range(len(x_test)):
        t = ' '.join([index_to_word[i] for index in x_test[i]])
        decoded.append(t)

    x_test = decoded
    
    x_train_dtm = dtmvector.fit_transform(x_train)
    x_train = tfidf_transformer.fit_transform(x_train_dtm)
    x_test_dtm = dtmvector.transform(x_test) #테스트 데이터를 DTM으로 변환
    x_test = tfidf_transformer.transform(x_test_dtm) #DTM을 TF-IDF 행렬로 변환
    
    return x_train, y_train, x_test, y_test
```

수정
``` python
def reuters_load_ml(num_words):
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=num_words, test_split=0.2)

    def decode_data(data):
        decoded = []
        for i in range(len(x_train)):
            t = ' '.join([index_to_word[index] for index in x_train[i]])
            decoded.append(t)
        return decoded

    x_train = decode_data(x_train)
    x_test = decode_data(x_test)

    dtmvector = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    
    x_train_dtm = dtmvector.fit_transform(x_train)
    x_train = tfidf_transformer.fit_transform(x_train_dtm)
    x_test_dtm = dtmvector.transform(x_test) #테스트 데이터를 DTM으로 변환
    x_test = tfidf_transformer.transform(x_test_dtm) #DTM을 TF-IDF 행렬로 변환
    
    return x_train, y_train, x_test, y_test
```

- [ ] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
- [ ] 코드가 간결한가요? 
 
 ----------------------------------------------

참고 링크 및 코드 개선
