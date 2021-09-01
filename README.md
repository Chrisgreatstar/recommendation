## Fundamentals of Recommender Systems
Implementation of [Recommender Systems](http://csse.szu.edu.cn/staff/panwk/recommendation/).

Dataset: [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)

Optimizer: Stochastic Gradient Descent (SGD)

See code example: [Factorizing Personalized Markov Chains](https://github.com/Chrisgreatstar/recommendation/blob/main/Recommendation%20with%20sequential%20feedback/Factorizing%20Personalized%20Markov%20Chains/implement.py) (using tensorflow: [Tensorflow Example](https://github.com/Chrisgreatstar/recommendation/blob/main/Recommendation%20with%20sequential%20feedback/Factorizing%20Personalized%20Markov%20Chains/tf_implement.py))


### Recommendation with explicit feedback (Multi-class feedback)

Evaluated by Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) as the distance between the real ratings and the predicted ratings. The calculation of MAE and RMSE as below:

![equation](https://latex.codecogs.com/gif.latex?MAE&space;=&space;\sum_{(u,&space;i,&space;r_{ui})&space;\in&space;R^{te}&space;}&space;|&space;r_{ui}&space;-&space;\widehat{r}_{ui}&space;|&space;/&space;|R^{te}|)

![equation](https://latex.codecogs.com/gif.latex?RMSE&space;=&space;\sqrt{\sum_{(u,&space;i,&space;r_{ui})&space;\in&space;R^{te}&space;}&space;(&space;r_{ui}&space;-&space;\widehat{r}_{ui}&space;)&space;^&space;2&space;/&space;|R^{te}|})

State of understanding and implementation of slides regarding this subject:

|                          Slide                          | Understanding | Implemented |
|:-------------------------------------------------------:|:-------------:|:-----------:|
|                     Average Filling                     |       1       |      1      |
|           Memory-Based Collaborative Filtering          |       1       |      1      |
|                   Matrix Factorization                  |       1       |      1      |
|                          SVD++                          |       1       |      1      |
|                  Factorization Machine                  |       0       |      0      |
| Matrix Factorization with Multiclass Preference Context |       1       |      1      |
|                          k-CoFi                         |       1       |      0      |

### Recommendation with implicit feedback (One-class feedback)

Evaluated by Ranking-Oriented Evaluation Metrics as the rationality of ranking, e.g. the precision denotes the proportion of recommended items in the test set. The full implementation of [Ranking Evaluation](https://github.com/Chrisgreatstar/recommendation/blob/main/utils/ranking_evaluation.py).

State of understanding and implementation of slides regarding this subject:

|                      Slide                     | Understanding | Implemented |
|:----------------------------------------------:|:-------------:|:-----------:|
|       Ranking-Oriented Evaluation Metrics      |       1       |      1      |
| Memory-Based One-Class Collaborative Filtering |       1       |      1      |
|          Bayesian Personalized Ranking         |       1       |      1      |
| Factored Item Similarity Models with RMSE Loss |       1       |      1      |
|  Factored Item Similarity Models with AUC Loss |       1       |      1      |
|     Matrix Factorization with Logistic Loss    |       1       |      1      |
|           Latent Dirichlet Allocation          |       0.5     |      0      |
|      Collaborative Denoising Auto-Encoders     |       0       |      0      |
|         Variational Auto-Encoders (VAE)        |       1       |      0      |
|  element-wise Alternative Least Squares (eALS) |       1       |      0      |

### Recommendation with sequential feedback
Take user-interacted items sorted by time sequence as input and evaluated by Ranking-Oriented Evaluation Metrics.

State of understanding and implementation of slides regarding this subject:

|                     Slide                     | Understanding | Implemented |
|:---------------------------------------------:|:-------------:|:-----------:|
|     Factorizing Personalized Markov Chains    |       1       |      1      |
|  Fusing Similarity Models with Markov Chains  |       0       |      0      |
|        Translation-based Recommendation       |       0       |      0      |
|    Self-Attentive Sequential Recommendation   |       1       |      0      |


