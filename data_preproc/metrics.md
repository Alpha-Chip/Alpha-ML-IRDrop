# Metrics 

In the work frame of the contest, two quality metrics were proposed: MAE (Mean Absolute Error) and F1 score (Chicco, Warrens & Jurman, 2021). The MAE score was considered the most important when assessing the IR drop prediction quality; its contribution to the overall quality score was set to 60%. The F1 score contributed 30%, and the remaining 10% was contributed by the computation speed. The MAE score evaluates the overall quality of the solution across the entire design. The F1 score evaluates the quality only in critical sections of the circuit with the maximum IR drop value. It is important for chip designers to know coordinates in the floorplan where the voltage drop is maximum. In these points it is necessary to solve the problem of large voltage drop. All other circuit locations have less IR drop, thus, calculations accuracy plays much smaller role there.

## MAE Score

MAE, or mean absolute error, is the average of the absolute difference between the predicted and actual value calculated for each example in the dataset. The goal is to minimize the MAE score.

### Example: 

let’s take a 2×2 matrix for clarity. Let’s assume we got the following IR drop values:
```
3.1 7.4
3.2 1.1
```
While actual values of IR drop are:
```
3.5 6.3
5.7 1.0
```
Then the MAE matrix will look as follows:
```
0.4 1.1
2.5 0.1
```
The MAE score after averaging will be equal to: 
```
MAE = 1.025
```
This is the average error value over the entire IR drop matrix.

## F1 Score

The F1 score is a binary classification metric. In this task, it uses 10% of the maximum IR drop of each circuit as the classification threshold. The formulas for calculating the F1 score are as follows:
`F1=(2*Precision*Recall)/(Precision+Recall)` where `Precision=TP/(TP+FP)`, `Recall=TP/(TP+FN)`.

On the IR drop map, the value 1 (True) marks areas where IR drop is higher than 90% of the maximum for the given circuit. We have to find such areas as accurately as possible. The more accurate results, the higher the F1 score.

According to the example from the MAE section, the maximum IR drop value for the matrix is 6.3. If take 90% of it, then the value would be 5.67. Thus, all the values greater than 5.67 in both the predicted matrix and the real matrix will be set to 1, and the remaining values will be set to 0.
So, the hotspot matrix for the predicted matrix will be:
```
0 1
0 0
```
For the real matrix:
```
0 1
1 0
```
Thus:
```
TP = 1, TN = 2, FP = 0, FN = 1.
```

Using the formulas, the F1 score can be calculated as:
```
Precision = 1 / (1+0) = 1
Recall = 1 ⁄ (1+1) = 0.5
F1 = (2*1*0.5) ⁄ (1+0.5) = 0.67
```