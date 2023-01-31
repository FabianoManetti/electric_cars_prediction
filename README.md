# Prediction of Energy Consumption of Electric Cars

<div align="center">
<img src="https://img.shields.io/badge/R-276DC3?style=for-the-badge&logo=r&logoColor=white"><img>
<img src="https://img.shields.io/badge/RStudio-75AADB?style=for-the-badge&logo=RStudio&logoColor=white"><img>
</div>

**This is part of the first training course of https://www.datascienceacademy.com.br/ Data Scientist program.**


<center><img src="electric_car.png"></center><br>


# Definition

The current project consists of creating a machine learning model to
predict the **energy consumption of electric cars**.

# Dataset

The public dataset, available on
<https://data.mendeley.com/datasets/tb9yrptydn/2>, lists attributes of
electric passenger cars collected on specialized websites in Poland.

| **Feature Name**              | **Description**                         |
|-------------------------------|-----------------------------------------|
| **Car full name**             | Car full name                           |
| **Make**                      | Car brand                               |
| **Model**                     | Car model                               |
| **Minimal price**             | Car minimal price (gross PLN)           |
| **Engine Power**              | Engine power (kM)                       |
| **Type of brakes**            | Type of brakes                          |
| **Drive type**                | Car drive type                          |
| **Battery capacity**          | Car battery capacity (kWh)              |
| **Range**                     | Car range (WLTP km)                     |
| **Wheelbase**                 | Car wheelbase distance (cm)             |
| **Length**                    | Car length (cm)                         |
| **Width**                     | Car width (cm)                          |
| **Height**                    | Car height (cm)                         |
| **Minimal empty weight**      | Car minimal empty weight (kg)           |
| **Permissable gross weight**  | Permissable gross weight (kg)           |
| **Maximum load capacity**     | Car maximum load capacity (kg)          |
| **Number of seat**            | Number of seats                         |
| **Number of doors**           | Number of doors                         |
| **Tire size**                 | Car tire size (in)                      |
| **Maximum speed**             | Maximum speed achieved by the car (kph) |
| **Boot capacity**             | Car boot capacity (VDA l)               |
| **Acceleration 0-100 kph**    | Time to reach 100 kph from 0 (s)        |
| **Maximum DC charging power** | Maximum charging power DC (kW)          |
| **Energy consumption**        | Mean energy consumption (kWh/100 km)    |

# First Analysis: Dropping missing values

## Exploratory Analysis

### Mean of `Consumption` by categorical features

* Relation between `Consumption` and `Brand`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-1.png"></center><br>

By this graph plot, we could categorize the top 10 `Brands` with the
**highest mean of `Consumption`** and verify that there are significant
difference among them. However, due to the reduced number of
observations in this dataframe, we should use this information with
caution.

* Relation between `Consumption` and `Type_Brakes`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-2.png"></center><br>

The difference in the median of the two `Types_Brakes`, in terms of
`Consumption` **doesn’t seem to be significant**, even though the
difference in the pattern of the data (again, we have to consider that
we have more examples of one category).

* Relation between `Consumption` and `Drive_Type`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-3.png"></center><br>

We can identify a clear possibility of the 4WD `Drive_Type` to be
**statistically different** from the other two categories, indicating
that this feature might be a good predictor for the model.

### 7.1.2 Mean of `Consumption` by numerical features

* Relation between `Consumption` and `Price`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-1.png"></center><br>

In general, the **higher the `Price`, the higher the mean of
`Consumption`**. For smaller Prices though, there isn’t a clear
tendency. The relation between `Price` and the other numerical features
will be plot in the sequence.

* Relation between `Consumption` and `Power`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-2.png"></center><br>

`Power` has a **positive correlation** with the mean of `Consumption`.

* Relation between `Consumption` and `Torque`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-3.png"></center><br>

Similarly to `Power`, the graph plot indicates that **the higher the
`Torque`, the higher the `Consumption`**. In fact, `Power`and `Torque`
might have a high correlation between each other and this relation will
be examined later.

* Relation between `Consumption` and `Battery_Capacity`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-4.png"></center><br>

It’s possible to see a **positive correlation between `Battery_Capacity`
and `Consumption`**.

* Relation between `Consumption` and `Range`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-5.png"></center><br>

In this case, there isn’t a **clear tendency between the feature `Range`
and the car `Consumption`**.

* Relation between `Consumption`, `Wheelbase`, `Length`, `Width` and
  `Height`
  
<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-6.png"></center><br>

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-7.png"></center><br> 

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-8.png"></center><br> 

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-9.png"></center><br> 
 
 It seems that, related to the car dimensions, **the higher that
dimension, the higher the car `Consumption` (except for the feature
`Height`)**.

* Relation between `Consumption`, `Minimal_Weight` and `Gross_Weight`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-10.png"></center><br>

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-11.png"></center><br> 

As expected, **higher values of `Weight` are associated to higher
`Consumption`**. Both graph plots have similar behavior, for this reason
the relation between these two features, along with the other
dimensions, will be examined later.

* Relation between `Consumption` and `Load_Capacity`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-12.png"></center><br>

Once again, we observe a **positive correlation** between the
`Load_Capacity` of a car and its `Consumption`.

* Relation between `Consumption`, `Seats`, `Doors` and `Tire_Size`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-13.png"></center><br>

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-14.png"></center><br> 

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-15.png"></center><br> 

Among these three features, only `Tire_Size` **might be a good
predictor** for `Consumption`.

* Relation between `Consumption`, `Max_Speed` and `Acceleration`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-16.png"></center><br>

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-18.png"></center><br> 

Both features seem to have **good positive correlation with
`Consumption`**. Nevertheless it’s important to consider a
multicollinearity between these two variables, since they represent
essencially the same result.

* Relation between `Consumption` and `Boot_Capacity`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-17.png"></center><br>

Once again, it’s possible to affirm that **the higher the
`Boot_Capacity`, the higher the car `Consumption`**.

* Relation between `Consumption` and `Maximum_DC_Charging Power`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-19.png"></center><br>

Although the feature has apperentaly **good positive correlation with
`Consumption`**, it’s important to extend our study on this variable,
specially comparing it to `Battery_Capacity` in order to avoid eventual
interference in the quality of our prediction model.

* Histogram + Boxplot of `Consumption`

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-27-1.png"></center><br>

The histogram of `Consumption` indicates that this features **does not
follow a normal distribution**. The bulk of data are concentrated on
**smaller values of `Consumption`**, even though both, histogram and
boxplot, shows a distortion towards highest values.

## Splitting data

From our exploratory analysis, we decided to choose the following
features as input to our regression prediction model: `Power`, `Torque`,
`Drive_Type`, `Battery_Capacity`, `Gross_Weight`, `Load_Capacity`,
`Tire_Size`, `Max_Speed` and `Boot_Capacity`. As we previously
concluded, the other features either have multicollinearity or were
already represented by the chosen ones.

## Standardizing data

Before the process of training we need to perform stardardization of the
features, so that we **prevent features with wider ranges from
dominating others**. For this purpose, we’ll take the mean and standard
deviation from the training set and use them to stardardize both, the
training and test set.

## Choosing and training models

For our regression project, we decided to test the following machine
learning models: **Linear Regression**, **Ridge Regression**, **Random
Forest** and **XGBoost**.

For our first running, we will consider all the previously selected
features, which will trained across a 5-fold-cross validation method (in
order to avoid randomness of evaluation).

## Evaluating models

Our main objective for this project is to deliver a model that will be
used to predict the Energy Consumption of electrical cars.

In this case, we are concerned in reducing the error of our model. Three
metrics will be used to evaluate the result: **R²** or coefficient of
determination, which is the proportion of the variance for a dependent
variable that is explained by independend variables; **MAE** or mean
absolute error, which is the average absolute error between actual and
predicted values; **RMSE** or root mean square error, which is the
starndard deviation of the residuals (prediction errors).

All the models tested presented similar values for R² metric, while
XGBoost and Random Forest showed slight better values for RMSE and MAE
metrics, for the test set. However, when looking at the training results
we see that Randon Forest model didn’t present the same performance.
Ridge Regression was able to perform better for the training set but
didn’t improve the results for the test set compared to Linear
Regression not regularized. As for XGBoost, it did achieve a set of
parameters that elevated the performance for the training and test set.
XGBoost requires, however, more computational power and brings
complexity to the model; Linear Regression model, on the other hand, is
a simpler and faster algorithm that showed good performance and
consistence for our dataset, this is why we decided to follow with the
**Linear Regression** model.

## Ranking features by importance

<center><img src="electric_cars_prediction_files/figure-gfm/unnamed-chunk-44-1.png"></center><br>

As we can see, the feature `Boot_Capacity` seems to have a smaller
effect on the prediction model. For this reason, we will not consider it
in our final model.

## Optmizing model

We were able to keep up our performance while we reduced dimensionality
of the final model.

# Second Analysis: Inputting missing values

