# CarND-13-Localization-Bayes-Markov-Localization
Udacity Self-Driving Car Engineer Nanodegree: Markov-Localization.

## Goal

- We aim to estimate state beliefs without the need to carry our entire observation history. 

- We will accomplish this by manipulating our posterior, obtaining a recursive state estimator. 

- For this to work, we must demonstrate that our current belief can be expressed by the belief one step earlier, then use new data to update only the current belief.

<img src="https://github.com/ChenBohan/Auto-Car-Sensor-Fusion-02-Lidar-and-Radar-Fusion/blob/master/readme_img/Overview%20of%20the%20Kalman%20Filter%20Algorithm%20Map.png" width = "70%" height = "70%" div align=center />

## Summary

We have accomplished a lot in this lesson.

- Starting with the generalized form of Bayes Rule we expressed our posterior, the belief of x at t as `η` (normalizer) multiplied with the observation model and the motion model.

- We expressed the motion model as a recursive state estimator using the Markov assumption and the law of total probability, resulting in a model that includes our belief at t – 1 and our transition model.

- We simplified the observation model using the Markov assumption to determine the probability of z at time t, given only x at time t, and the map.

- Finally we derived the general Bayes Filter for Localization (Markov Localization) by expressing our belief of x at t as a simplified version of our original posterior expression (top equation), `η` multiplied by the simplified observation model and the motion model. Here the motion model is written as `bel`, a prediction model.

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/21-i-finalize-the-bayes-localization-filter.00-01-00-15.still001.png" width = "70%" height = "70%" div align=center />

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

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/Total%20Probability%20and%20Markov%20Assumption.png" width = "70%" height = "70%" div align=center />

### Markov Assumption

#### Motion Model

- we (hypothetically) know in which state the system is at time step t-1, the past observations `z_{1:t-1}` and controls `u_{1:t-1}` would not provide us additional information to estimate the posterior for `x_t`, because they were already used to estimate `x_{t-1}`. 

- `u_t` is “in the future” with reference to x_{t-1}, `u_t` does not tell us much about `x_{t-1}`.

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/Markov%20Assumption.png" width = "70%" height = "70%" div align=center />


#### Observation Model

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/20-i-markov-assumption-for-observation-model-first-try.00-01-18-09.still003.png" width = "70%" height = "70%" div align=center />


### Recursive Structure

#### Motion Model

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/14-l-explain-recursive-structure-.00-01-55-15.still004.png" width = "70%" height = "70%" div align=center />

## Implementation of Motion Model

- For each `x_{t}`

  - Calculate the transition probability for each potential value `x_{t-1}` 
  
  - Calculate the discrete motion model probability by multiplying the transition model probability by the belief state (prior) for `x_{t-1}`

- Return total probability (sum) of each discrete probability

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

## Implementation of Observation Model

### Observation Model

- For each pseudo position x:

  - For each observation:
    - determine if a pseudo range vector exists for the current pseudo position x
    - if the vector exists, extract and store the minimum distance, element 0 of the sorted vector, and remove that element (so we don't re-use it). This will be passed to norm_pdf
    - if the pseudo range vector does not exist, pass the maximum distance to norm_pdf
    - use norm_pdf to determine the observation model probability
    - return the total probability
    
```python
// calculates likelihood prob term based on landmark proximity
float observation_model(vector<float> landmark_positions, 
                        vector<float> observations, vector<float> pseudo_ranges, 
                        float distance_max, float observation_stdev) {
  // initialize observation probability
  float distance_prob = 1.0f;

  // run over current observation vector
  for (int z=0; z< observations.size(); ++z) {
    // define min distance
    float pseudo_range_min;
        
    // check, if distance vector exists
    if (pseudo_ranges.size() > 0) {
      // set min distance
      pseudo_range_min = pseudo_ranges[0];
      // remove this entry from pseudo_ranges-vector
      pseudo_ranges.erase(pseudo_ranges.begin());
    } else {  // no or negative distances: set min distance to a large number
        pseudo_range_min = std::numeric_limits<const float>::infinity();
    }

    // estimate the probability for observation model, this is our likelihood 
    distance_prob *= Helpers::normpdf(observations[z], pseudo_range_min,
                                      observation_stdev);
  }
  
  return distance_prob;
}
```

### pseudo_range_estimator

- For each pseudo position x:
  - For each landmark position:
    - determine the distance between each pseudo position x and each landmark position
    - if the distance is positive (landmark is forward of the pseudo position) push the distance to the pseudo range vector
    - sort the pseudo range vector in ascending order
    - return the pseudo range vector

```python
vector<float> pseudo_range_estimator(vector<float> landmark_positions, 
                                     float pseudo_position) {
  // define pseudo observation vector
  vector<float> pseudo_ranges;
            
  // loop over number of landmarks and estimate pseudo ranges
  for (int l=0; l< landmark_positions.size(); ++l) {
    // estimate pseudo range for each single landmark 
    // and the current state position pose_i:
    float range_l = landmark_positions[l] - pseudo_position;

    // check if distances are positive: 
    if (range_l > 0.0f) {
      pseudo_ranges.push_back(range_l);
    }
  }

  // sort pseudo range vector
  sort(pseudo_ranges.begin(), pseudo_ranges.end());

  return pseudo_ranges;
}
```

## Implementation of Full Filter

Implement the Bayes' localization filter by first initializing priors, then doing the following within each time step:

- extract sensor observations

  - for each pseudo-position:
    - get the motion model probability
    - determine pseudo ranges
    - get the observation model probability
    - use the motion and observation model probabilities to calculate the posterior probability
  - normalize posteriors (see helpers.h for a normalization function)
  - update priors (priors --> posteriors)

```python
// step through each pseudo position x (i)
for (unsigned int i = 0; i < map_size; ++i) {
  float pseudo_position = float(i);

  /**
   * TODO: get the motion model probability for each x position
   */
  float motion_prob = motion_model(pseudo_position, movement_per_timestep,
                                   priors, map_size, control_stdev);
  /**
   * TODO: get pseudo ranges
   */
  vector<float> pseudo_ranges = pseudo_range_estimator(landmark_positions, 
                                                       pseudo_position);

  /**
   * TODO: get observation probability
   */
  float observation_prob = observation_model(landmark_positions, observations, 
                                             pseudo_ranges, distance_max, 
                                             observation_stdev);

  /**
   * TODO: calculate the ith posterior
   */
  posteriors[i] = motion_prob * observation_prob; 
} 

/**
 * TODO: normalize
 */
posteriors = Helpers::normalize_vector(posteriors);

/**
 * TODO: update
 */
priors = posteriors;
}

```

## Bayes Filter Theory Summary

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/22-l-bayes-filter-theory-summary.00-00-22-29.still001.png" width = "70%" height = "70%" div align=center />

- The Bayes Localization Filter Markov Localization is a general framework for recursive state estimation.

- That means this framework allows us to use the previous state (state at t-1) to estimate a new state (state at t) using only current observations and controls (observations and control at t), rather than the entire data history (data from 0:t).

<img src="https://github.com/ChenBohan/CarND-13-Localization-Bayes-Markov-Localization/blob/master/readme_img/22-l-bayes-filter-theory-summary.00-00-52-03.still002.png" width = "70%" height = "70%" div align=center />

- The motion model describes the prediction step of the filter while the observation model is the update step.

- The state estimation using the Bayes filter is dependent upon the interaction between prediction (motion model) and update (observation model steps) and all the localization methods discussed so far are realizations of the Bayes filter.
