# CarND-13-Localization-Markov-Localization
Udacity Self-Driving Car Engineer Nanodegree: Markov-Localization.

## Goal

- We aim to estimate state beliefs without the need to carry our entire observation history. 

- We will accomplish this by manipulating our posterior, obtaining a recursive state estimator. 

- For this to work, we must demonstrate that our current belief can be expressed by the belief one step earlier, then use new data to update only the current belief.

pic

## Bayes Rule

- Likehood -> Obervation model

  - describes the probability distribution of the observation vector.
  
- Prior -> Motion model

  - describes the probability distribution of `xt` given all observation from `1` to `t-1`.
  
## Total Probability

pic

## Markov Assumption

- we (hypothetically) know in which state the system is at time step t-1, the past observations `z_{1:t-1}` and controls `u_{1:t-1}` would not provide us additional information to estimate the posterior for `x_t`, because they were already used to estimate `x_{t-1}`. 

- `u_t` is “in the future” with reference to x_{t-1}, `u_t` does not tell us much about `x_{t-1}`.
