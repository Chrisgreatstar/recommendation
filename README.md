## Fundamentals of Recommender Systems
Implementation of [Recommender Systems](http://csse.szu.edu.cn/staff/panwk/recommendation/).

Including all slides of [Recommendation with explicit feedback], machine learning based methods of [Recommendation with implicit feedback] and the slide [Factorizing Personalized Markov Chains] of [Recommendation with sequential feedback].

### Recommendation with explicit feedback
Explicit feedback means data format based on numerical ratings (or said Multi-class feedback).

Evaluated by Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as the distance between the real rating and the predicted rating.

$MAE = \sum_{(u, i, r_{ui}) \in   R^{te}  }     | r_{ui} - \widehat{r}_{ui} | / |R^{te}|$

$RMSE = \sqrt{\sum_{(u, i, r_{ui}) \in   R^{te}  }     ( r_{ui} - \widehat{r}_{ui} ) ^ 2 / |R^{te}|}$

### Recommendation with implicit feedback
As One-class feedback.

Evaluated by Ranking-Oriented Evaluation Metrics as the rationality of ranking, e.g. the precision denotes the proportion of recommended items in the test set.

The full implementation of [Ranking Evaluation](https://github.com/Chrisgreatstar/recommendation/blob/main/utils/ranking_evaluation.py).


### Recommendation with sequential feedback
Take user-interacted items sorted by time sequence as input.

Evaluated by Ranking-Oriented Evaluation Metrics.

code example: [Factorizing Personalized Markov Chains](https://github.com/Chrisgreatstar/recommendation/blob/main/Recommendation%20with%20sequential%20feedback/Factorizing%20Personalized%20Markov%20Chains/implement.py)







