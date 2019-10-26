# Readme
### MVP
* Sequential conv net
* Train frames independently


### Problems
* When we leave the expert trajectory, the model doesn't know what to do
    * Definition of failure:
        * When you are off the track, i.e. not collecting points
* Minor errors add up <-> Don't have iid samples
    * The state error is not independent of the previous state's error


### Extensions - improvements
* **Data Augmentation**
    * MIRROR! because there are mostly left turns, rotate, decolorise
    * More examples when cars got off the road
        * Increase the curve examples/ decrease the straight line examples
        * Watch the agent and take over when it makes a mistake
    * Vary the tracks by changing track seeds
    * **DAGGER**
        * Queries the expert for an aggregate dataset
        * Expert takes over
    * Find datasets online
    * Find technical solution for others to play
* **Image preprocessing**
    * Write your own data loader
    * Use greyscale images
    * Crop 
        * Make sure that speed, gyro, ... are connected to the fc layer
    
  
### Other ideas
* Each frame may be classified differently -> May need to take moving averages/ Majority votes on deciding the action
* He suggests to start with MSE Loss
* Find last years results/ code to compare
* Could discretise action space -> steering angle in bins


--------------------------
### Environment 
* Agent gets 96x96 low res image, not the same you see when playing
* **Reward**
    * $R=N_{visited_tile} * \frac{1000}{N_{all\_tiles}}$ given a fixed number of frames
    * $R =R-100$ if the car gets too far away. i.e. crashes
    * Maximum reward is to come back to the beginning. Consistent score of 900 is considered a solution. 
* No friction
* A bunch of colors on the bottom of the screen
* Use their car_racing.py to start the game/ use their gym zip from Ilias (see ppt)


----------------
### Workflow - Cluster
0. Get a cluster account -> ssh in
1. Copy the singularity image to the cluster
2. Copy your code to the cluster
3. Run


-------------
### Things to check
* When using regression, make sure the range of NN matches the variable range


-------------
### Provided Code
* `record_imitations` function -> add to data folder, user same naming conventions
* `extract_sensor_values` for access of the true speed, gyro etc
* `calculate_score_for_leaderboard` will be used to evaluate, 

--------------
## Questions
* Why does `scores_to_action` get a list of scores, but the only evaluates the first score?
* Cross entropy loss
* Inverse reinforcement learning