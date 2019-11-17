# CarND-13-Localization-Bayes-Markov-Localization
Udacity Self-Driving Car Engineer Nanodegree: Markov-Localization.

## Goal

- We aim to estimate state beliefs without the need to carry our entire observation history. 

- We will accomplish this by manipulating our posterior, obtaining a recursive state estimator. 

- For this to work, we must demonstrate that our current belief can be expressed by the belief one step earlier, then use new data to update only the current belief.

pic

## Summary

We have accomplished a lot in this lesson.

- Starting with the generalized form of Bayes Rule we expressed our posterior, the belief of x at t as `η` (normalizer) multiplied with the observation model and the motion model.

- We expressed the motion model as a recursive state estimator using the Markov assumption and the law of total probability, resulting in a model that includes our belief at t – 1 and our transition model.

- We simplified the observation model using the Markov assumption to determine the probability of z at time t, given only x at time t, and the map.

- Finally we derived the general Bayes Filter for Localization (Markov Localization) by expressing our belief of x at t as a simplified version of our original posterior expression (top equation), `η` multiplied by the simplified observation model and the motion model. Here the motion model is written as `bel`, a prediction model.

pic

## Simplfy Rule

### Bayes Rule

- Likehood -> Obervation model

  - describes the probability distribution of the observation vector.
  
- Prior -> Motion model

  - describes the probability distribution of `xt` given all observation from `1` to `t-1`.
  
  - Calculating the probability that the vehicle is now at a given location `x_tx`.
  
  - For each possible prior location in that list, `x_{t-1}`, the summation yields the total probability that the vehicle really did start at that prior location and that it wound up at `x_t`.
  
### Total Probability

#### Motion Model

pic

### Markov Assumption

#### Motion Model

- we (hypothetically) know in which state the system is at time step t-1, the past observations `z_{1:t-1}` and controls `u_{1:t-1}` would not provide us additional information to estimate the posterior for `x_t`, because they were already used to estimate `x_{t-1}`. 

- `u_t` is “in the future” with reference to x_{t-1}, `u_t` does not tell us much about `x_{t-1}`.

pic

#### Observation Model

pic

### Recursive Structure

#### Motion Model

pic


## Motion Model

```python
// implement the motion model: calculates prob of being at 
// an estimated position at time t
float motion_model(float pseudo_position, float movement, vector<float> priors,
                   int map_size, int control_stdev) {
  // initialize probability
  float position_prob = 0.0f;

  // loop over state space for all possible positions x (convolution):
  for (float j=0; j< map_size; ++j) {
    float next_pseudo_position = j;
    // distance from i to j
    float distance_ij = pseudo_position-next_pseudo_position;

    // transition probabilities:
    float transition_prob = Helpers::normpdf(distance_ij, movement, 
                                             control_stdev);
    // estimate probability for the motion model, this is our prior
    position_prob += transition_prob*priors[j];
  
  return position_prob;
}

```
