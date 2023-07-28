----------------------------------------------

- 코더 : 남희정
- 리뷰어 : 소용현

----------------------------------------------

PRT(PeerReviewTemplate)

- [o] 코드가 정상적으로 동작하고 주어진 문제를 해결했나요?
![image](https://github.com/eveningwalk/AIFFEL_Project/assets/100551891/203dc595-6c9d-4e50-8dc7-ea957cdcdaed)
3단계 학습 완료하였습니다.
- [o] 주석을 보고 작성자의 코드가 이해되었나요?
- [x] 코드가 에러를 유발할 가능성이 있나요?
- [o] 코드 작성자가 코드를 제대로 이해하고 작성했나요? (직접 인터뷰해보기)
 인터뷰 결과 제대로 이해하고 작성하였습니다.
- [o] 코드가 간결한가요?
허깅페이스를 이용하여 간결하게 작성되었습니다.
 ```
trainer = RewardModelTrainer(model=model,
                             strategy=NaiveStrategy(),
                             optim=Adam(model.parameters(), lr=5e-5),
                             train_dataset=train_dataset,
                             eval_dataset=eval_dataset,
                             batch_size=4,
                             #max_epochs=1)
                             max_epochs= max_epochs_rm)
```

 ----------------------------------------------

참고 링크 및 코드 개선
