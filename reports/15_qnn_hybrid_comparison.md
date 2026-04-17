# QNN Hybrid Reproduction Comparison

- Data: `data\processed\pretrain_dataset.csv`
- Split: temporal holdout (last 20% by date)
- Train/Test: 9676/2419

| model      |   accuracy |   precision_macro |   recall_macro |   f1_macro | notes                                        |
|:-----------|-----------:|------------------:|---------------:|-----------:|:---------------------------------------------|
| LR         |   0.484498 |          0.444535 |       0.450113 |   0.444975 | temporal holdout                             |
| KNN        |   0.461761 |          0.407813 |       0.414254 |   0.402197 | temporal holdout                             |
| BP-MLP     |   0.397685 |          0.372162 |       0.372444 |   0.371895 | temporal holdout                             |
| Hybrid-QNN |   0.335676 |          0.336988 |       0.336185 |   0.331906 | q=4, layers=1, hidden=24, best_val_f1=0.3341 |

## Detailed Reports

### KNN
```
              precision    recall  f1-score   support

           A     0.4557    0.4299    0.4424       777
           D     0.2500    0.1388    0.1785       605
           H     0.5178    0.6741    0.5857      1037

    accuracy                         0.4618      2419
   macro avg     0.4078    0.4143    0.4022      2419
weighted avg     0.4309    0.4618    0.4378      2419

```
### LR
```
              precision    recall  f1-score   support

           A     0.4788    0.5238    0.5003       777
           D     0.2854    0.2132    0.2441       605
           H     0.5694    0.6133    0.5905      1037

    accuracy                         0.4845      2419
   macro avg     0.4445    0.4501    0.4450      2419
weighted avg     0.4693    0.4845    0.4749      2419

```
### BP-MLP
```
              precision    recall  f1-score   support

           A     0.3884    0.3719    0.3800       777
           D     0.2504    0.2314    0.2405       605
           H     0.4776    0.5140    0.4951      1037

    accuracy                         0.3977      2419
   macro avg     0.3722    0.3724    0.3719      2419
weighted avg     0.3921    0.3977    0.3945      2419

```
### Hybrid-QNN
```
              precision    recall  f1-score   support

           A     0.3277    0.3475    0.3373       777
           D     0.2472    0.3322    0.2835       605
           H     0.4361    0.3288    0.3749      1037

    accuracy                         0.3357      2419
   macro avg     0.3370    0.3362    0.3319      2419
weighted avg     0.3540    0.3357    0.3400      2419

```