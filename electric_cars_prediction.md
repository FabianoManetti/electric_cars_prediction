Electric Car Energy Consumption Prediction
================
Fabiano Manetti

![](electric_car.png)

# 1. Definition

The current project consists of creating a machine learning model to
predict the **energy consumption of electric cars**.

# 2. Dataset

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

# 3. Setting working directory

``` r
setwd("C:/Users/fabia/OneDrive/Área de Trabalho/Arquivos/Data_Science/DSA/Big_Data_R_Azure/electric_cars_prediction")
getwd()
```

    ## [1] "C:/Users/fabia/OneDrive/Área de Trabalho/Arquivos/Data_Science/DSA/Big_Data_R_Azure/electric_cars_prediction"

# 4. Importing libraries

``` r
library('ggplot2', quietly = T)
library("readxl", quietly = T)
library("data.table", quietly = T)
library("dplyr", quietly = T)
library('gridExtra', quietly = T)
library('corrplot', quietly = T)
library('nortest', quietly = T)
library('caret', quietly = T)
library('caTools', quietly = T)
library('fastDummies', quietly = T)
library('standardize', quietly = T)
```

# 5. Reading dataset

``` r
dataset <- read_excel('FEV-data-Excel.xlsx')
```

``` r
sample(dataset)
```

    ## # A tibble: 53 × 25
    ##    Heigh…¹ Maxim…² Engin…³ Minim…⁴ mean …⁵ Type …⁶ Maxim…⁷ Model Minim…⁸ Boot …⁹
    ##      <dbl>   <dbl>   <dbl>   <dbl>   <dbl> <chr>     <dbl> <chr>   <dbl>   <dbl>
    ##  1    163.     150     360    2565    24.4 disc (…     640 e-tr…  345700     660
    ##  2    163.     150     313    2445    23.8 disc (…     670 e-tr…  308400     660
    ##  3    163.     150     503    2695    27.6 disc (…     565 e-tr…  414900     660
    ##  4    162.     150     313    2445    23.3 disc (…     640 e-tr…  319700     615
    ##  5    162.     150     360    2595    23.8 disc (…     670 e-tr…  357000     615
    ##  6    162.     150     503    2695    27.2 disc (…     565 e-tr…  426200     615
    ##  7    157       50     170    1440    13.1 disc (…     440 i3     169700     260
    ##  8    159       50     184    1460    14.3 disc (…     440 i3s    184200     260
    ##  9    167.     150     286    2260    18.8 disc (…     540 iX3    282900     510
    ## 10    152.     100     136    1541    NA   disc (…     459 ë-C4   125000     380
    ## # … with 43 more rows, 15 more variables: `Length [cm]` <dbl>,
    ## #   `Range (WLTP) [km]` <dbl>, `Wheelbase [cm]` <dbl>, `Tire size [in]` <dbl>,
    ## #   `Permissable gross weight [kg]` <dbl>, `Drive type` <chr>,
    ## #   `Acceleration 0-100 kph [s]` <dbl>, `Maximum speed [kph]` <dbl>,
    ## #   `Battery capacity [kWh]` <dbl>, `Number of doors` <dbl>,
    ## #   `Car full name` <chr>, `Number of seats` <dbl>,
    ## #   `Maximum torque [Nm]` <dbl>, Make <chr>, `Width [cm]` <dbl>, and …

# 6. Data Exploration

## 6.1 Summary

``` r
dim(dataset)
```

    ## [1] 53 25

``` r
str(dataset)
```

    ## tibble [53 × 25] (S3: tbl_df/tbl/data.frame)
    ##  $ Car full name                         : chr [1:53] "Audi e-tron 55 quattro" "Audi e-tron 50 quattro" "Audi e-tron S quattro" "Audi e-tron Sportback 50 quattro" ...
    ##  $ Make                                  : chr [1:53] "Audi" "Audi" "Audi" "Audi" ...
    ##  $ Model                                 : chr [1:53] "e-tron 55 quattro" "e-tron 50 quattro" "e-tron S quattro" "e-tron Sportback 50 quattro" ...
    ##  $ Minimal price (gross) [PLN]           : num [1:53] 345700 308400 414900 319700 357000 ...
    ##  $ Engine power [KM]                     : num [1:53] 360 313 503 313 360 503 170 184 286 136 ...
    ##  $ Maximum torque [Nm]                   : num [1:53] 664 540 973 540 664 973 250 270 400 260 ...
    ##  $ Type of brakes                        : chr [1:53] "disc (front + rear)" "disc (front + rear)" "disc (front + rear)" "disc (front + rear)" ...
    ##  $ Drive type                            : chr [1:53] "4WD" "4WD" "4WD" "4WD" ...
    ##  $ Battery capacity [kWh]                : num [1:53] 95 71 95 71 95 95 42.2 42.2 80 50 ...
    ##  $ Range (WLTP) [km]                     : num [1:53] 438 340 364 346 447 369 359 345 460 350 ...
    ##  $ Wheelbase [cm]                        : num [1:53] 293 293 293 293 293 ...
    ##  $ Length [cm]                           : num [1:53] 490 490 490 490 490 ...
    ##  $ Width [cm]                            : num [1:53] 194 194 198 194 194 ...
    ##  $ Height [cm]                           : num [1:53] 163 163 163 162 162 ...
    ##  $ Minimal empty weight [kg]             : num [1:53] 2565 2445 2695 2445 2595 ...
    ##  $ Permissable gross weight [kg]         : num [1:53] 3130 3040 3130 3040 3130 ...
    ##  $ Maximum load capacity [kg]            : num [1:53] 640 670 565 640 670 565 440 440 540 459 ...
    ##  $ Number of seats                       : num [1:53] 5 5 5 5 5 5 4 4 5 5 ...
    ##  $ Number of doors                       : num [1:53] 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ Tire size [in]                        : num [1:53] 19 19 20 19 19 20 19 20 19 16 ...
    ##  $ Maximum speed [kph]                   : num [1:53] 200 190 210 190 200 210 160 160 180 150 ...
    ##  $ Boot capacity (VDA) [l]               : num [1:53] 660 660 660 615 615 615 260 260 510 380 ...
    ##  $ Acceleration 0-100 kph [s]            : num [1:53] 5.7 6.8 4.5 6.8 5.7 4.5 8.1 6.9 6.8 9.5 ...
    ##  $ Maximum DC charging power [kW]        : num [1:53] 150 150 150 150 150 150 50 50 150 100 ...
    ##  $ mean - Energy consumption [kWh/100 km]: num [1:53] 24.4 23.8 27.6 23.3 23.9 ...

``` r
class(dataset)
```

    ## [1] "tbl_df"     "tbl"        "data.frame"

``` r
summary(dataset)
```

    ##  Car full name          Make              Model          
    ##  Length:53          Length:53          Length:53         
    ##  Class :character   Class :character   Class :character  
    ##  Mode  :character   Mode  :character   Mode  :character  
    ##                                                          
    ##                                                          
    ##                                                          
    ##                                                          
    ##  Minimal price (gross) [PLN] Engine power [KM] Maximum torque [Nm]
    ##  Min.   : 82050              Min.   : 82.0     Min.   : 160       
    ##  1st Qu.:142900              1st Qu.:136.0     1st Qu.: 260       
    ##  Median :178400              Median :204.0     Median : 362       
    ##  Mean   :246159              Mean   :269.8     Mean   : 460       
    ##  3rd Qu.:339480              3rd Qu.:372.0     3rd Qu.: 640       
    ##  Max.   :794000              Max.   :772.0     Max.   :1140       
    ##                                                                   
    ##  Type of brakes      Drive type        Battery capacity [kWh] Range (WLTP) [km]
    ##  Length:53          Length:53          Min.   : 17.60         Min.   :148.0    
    ##  Class :character   Class :character   1st Qu.: 40.00         1st Qu.:289.0    
    ##  Mode  :character   Mode  :character   Median : 58.00         Median :364.0    
    ##                                        Mean   : 62.37         Mean   :376.9    
    ##                                        3rd Qu.: 80.00         3rd Qu.:450.0    
    ##                                        Max.   :100.00         Max.   :652.0    
    ##                                                                                
    ##  Wheelbase [cm]   Length [cm]      Width [cm]     Height [cm]   
    ##  Min.   :187.3   Min.   :269.5   Min.   :164.5   Min.   :137.8  
    ##  1st Qu.:258.8   1st Qu.:411.8   1st Qu.:178.8   1st Qu.:148.1  
    ##  Median :270.0   Median :447.0   Median :180.9   Median :155.6  
    ##  Mean   :273.6   Mean   :442.5   Mean   :186.2   Mean   :155.4  
    ##  3rd Qu.:290.0   3rd Qu.:490.1   3rd Qu.:193.5   3rd Qu.:161.5  
    ##  Max.   :327.5   Max.   :514.0   Max.   :255.8   Max.   :191.0  
    ##                                                                 
    ##  Minimal empty weight [kg] Permissable gross weight [kg]
    ##  Min.   :1035              Min.   :1310                 
    ##  1st Qu.:1530              1st Qu.:1916                 
    ##  Median :1685              Median :2119                 
    ##  Mean   :1868              Mean   :2289                 
    ##  3rd Qu.:2370              3rd Qu.:2870                 
    ##  Max.   :2710              Max.   :3500                 
    ##                            NA's   :8                    
    ##  Maximum load capacity [kg] Number of seats Number of doors Tire size [in] 
    ##  Min.   : 290.0             Min.   :2.000   Min.   :3.000   Min.   :14.00  
    ##  1st Qu.: 440.0             1st Qu.:5.000   1st Qu.:5.000   1st Qu.:16.00  
    ##  Median : 486.0             Median :5.000   Median :5.000   Median :17.00  
    ##  Mean   : 520.5             Mean   :4.906   Mean   :4.849   Mean   :17.68  
    ##  3rd Qu.: 575.0             3rd Qu.:5.000   3rd Qu.:5.000   3rd Qu.:19.00  
    ##  Max.   :1056.0             Max.   :8.000   Max.   :5.000   Max.   :21.00  
    ##  NA's   :8                                                                 
    ##  Maximum speed [kph] Boot capacity (VDA) [l] Acceleration 0-100 kph [s]
    ##  Min.   :123.0       Min.   :171.0           Min.   : 2.500            
    ##  1st Qu.:150.0       1st Qu.:315.0           1st Qu.: 4.875            
    ##  Median :160.0       Median :425.0           Median : 7.700            
    ##  Mean   :178.2       Mean   :445.1           Mean   : 7.360            
    ##  3rd Qu.:200.0       3rd Qu.:558.0           3rd Qu.: 9.375            
    ##  Max.   :261.0       Max.   :870.0           Max.   :13.100            
    ##                      NA's   :1               NA's   :3                 
    ##  Maximum DC charging power [kW] mean - Energy consumption [kWh/100 km]
    ##  Min.   : 22.0                  Min.   :13.10                         
    ##  1st Qu.:100.0                  1st Qu.:15.60                         
    ##  Median :100.0                  Median :17.05                         
    ##  Mean   :113.5                  Mean   :18.99                         
    ##  3rd Qu.:150.0                  3rd Qu.:23.50                         
    ##  Max.   :270.0                  Max.   :28.20                         
    ##                                 NA's   :9

## 6.2 Changing column names

``` r
colnames(dataset) <- c("Car", "Brand", "Model", "Price", "Power", "Torque", "Type_Brakes",    "Drive_Type","Battery_Capacity", "Range", "Wheelbase", "Length", "Width", "Height", 
"Minimal_Weight", "Gross_Weight", "Load_Capacity", "Seats", "Doors", "Tire_Size","Max_Speed", "Boot_Capacity", "Acceleration_0-100", "Max_DC_charging", "Consumption")
```

## 6.3 Unique values of categorical columns

``` r
table(dataset$Brand)
```

    ## 
    ##          Audi           BMW       Citroën            DS         Honda 
    ##             6             3             2             1             2 
    ##       Hyundai        Jaguar           Kia         Mazda Mercedes-Benz 
    ##             3             1             4             1             2 
    ##          Mini        Nissan          Opel       Peugeot       Porsche 
    ##             1             3             2             2             4 
    ##       Renault         Skoda         Smart         Tesla    Volkswagen 
    ##             2             1             2             7             4

``` r
table(dataset$`Type_Brakes`)
```

    ## 
    ##        disc (front + rear) disc (front) + drum (rear) 
    ##                         45                          7

``` r
table(dataset$`Drive_Type`)
```

    ## 
    ## 2WD (front)  2WD (rear)         4WD 
    ##          24          11          18

## 6.4 Checking for missing values

``` r
column_names <- colnames(dataset)

missing_values <- c()

for (column in column_names){
  missing_values[column] <- sum(is.na(dataset[column]))
}

missing_values
```

    ##                Car              Brand              Model              Price 
    ##                  0                  0                  0                  0 
    ##              Power             Torque        Type_Brakes         Drive_Type 
    ##                  0                  0                  1                  0 
    ##   Battery_Capacity              Range          Wheelbase             Length 
    ##                  0                  0                  0                  0 
    ##              Width             Height     Minimal_Weight       Gross_Weight 
    ##                  0                  0                  0                  8 
    ##      Load_Capacity              Seats              Doors          Tire_Size 
    ##                  8                  0                  0                  0 
    ##          Max_Speed      Boot_Capacity Acceleration_0-100    Max_DC_charging 
    ##                  0                  1                  3                  0 
    ##        Consumption 
    ##                  9

We observe some **missing values** in our dataset. At this point, we
decided to create two analysis: in the first one, we will simply drop
the missing values; and then, we will try to input data according to
certain criteria.

# 7. First Analysis: Dropping missing values

``` r
df_1 <- copy(dataset)

df_1 <- na.omit(df_1)
```

``` r
dim(df_1)
```

    ## [1] 42 25

``` r
str(df_1)
```

    ## tibble [42 × 25] (S3: tbl_df/tbl/data.frame)
    ##  $ Car               : chr [1:42] "Audi e-tron 55 quattro" "Audi e-tron 50 quattro" "Audi e-tron S quattro" "Audi e-tron Sportback 50 quattro" ...
    ##  $ Brand             : chr [1:42] "Audi" "Audi" "Audi" "Audi" ...
    ##  $ Model             : chr [1:42] "e-tron 55 quattro" "e-tron 50 quattro" "e-tron S quattro" "e-tron Sportback 50 quattro" ...
    ##  $ Price             : num [1:42] 345700 308400 414900 319700 357000 ...
    ##  $ Power             : num [1:42] 360 313 503 313 360 503 170 184 286 136 ...
    ##  $ Torque            : num [1:42] 664 540 973 540 664 973 250 270 400 260 ...
    ##  $ Type_Brakes       : chr [1:42] "disc (front + rear)" "disc (front + rear)" "disc (front + rear)" "disc (front + rear)" ...
    ##  $ Drive_Type        : chr [1:42] "4WD" "4WD" "4WD" "4WD" ...
    ##  $ Battery_Capacity  : num [1:42] 95 71 95 71 95 95 42.2 42.2 80 50 ...
    ##  $ Range             : num [1:42] 438 340 364 346 447 369 359 345 460 320 ...
    ##  $ Wheelbase         : num [1:42] 293 293 293 293 293 ...
    ##  $ Length            : num [1:42] 490 490 490 490 490 ...
    ##  $ Width             : num [1:42] 194 194 198 194 194 ...
    ##  $ Height            : num [1:42] 163 163 163 162 162 ...
    ##  $ Minimal_Weight    : num [1:42] 2565 2445 2695 2445 2595 ...
    ##  $ Gross_Weight      : num [1:42] 3130 3040 3130 3040 3130 ...
    ##  $ Load_Capacity     : num [1:42] 640 670 565 640 670 565 440 440 540 450 ...
    ##  $ Seats             : num [1:42] 5 5 5 5 5 5 4 4 5 5 ...
    ##  $ Doors             : num [1:42] 5 5 5 5 5 5 5 5 5 5 ...
    ##  $ Tire_Size         : num [1:42] 19 19 20 19 19 20 19 20 19 17 ...
    ##  $ Max_Speed         : num [1:42] 200 190 210 190 200 210 160 160 180 150 ...
    ##  $ Boot_Capacity     : num [1:42] 660 660 660 615 615 615 260 260 510 350 ...
    ##  $ Acceleration_0-100: num [1:42] 5.7 6.8 4.5 6.8 5.7 4.5 8.1 6.9 6.8 8.7 ...
    ##  $ Max_DC_charging   : num [1:42] 150 150 150 150 150 150 50 50 150 100 ...
    ##  $ Consumption       : num [1:42] 24.4 23.8 27.6 23.3 23.9 ...
    ##  - attr(*, "na.action")= 'omit' Named int [1:11] 10 30 40 41 42 43 44 45 46 52 ...
    ##   ..- attr(*, "names")= chr [1:11] "10" "30" "40" "41" ...

``` r
summary(df_1)
```

    ##      Car               Brand              Model               Price       
    ##  Length:42          Length:42          Length:42          Min.   : 82050  
    ##  Class :character   Class :character   Class :character   1st Qu.:140650  
    ##  Mode  :character   Mode  :character   Mode  :character   Median :166945  
    ##                                                           Mean   :235066  
    ##                                                           3rd Qu.:316875  
    ##                                                           Max.   :794000  
    ##      Power           Torque       Type_Brakes         Drive_Type       
    ##  Min.   : 82.0   Min.   : 160.0   Length:42          Length:42         
    ##  1st Qu.:136.0   1st Qu.: 260.0   Class :character   Class :character  
    ##  Median :184.0   Median : 317.5   Mode  :character   Mode  :character  
    ##  Mean   :237.7   Mean   : 425.2                                        
    ##  3rd Qu.:313.0   3rd Qu.: 540.0                                        
    ##  Max.   :625.0   Max.   :1050.0                                        
    ##  Battery_Capacity     Range         Wheelbase         Length     
    ##  Min.   :17.60    Min.   :148.0   Min.   :187.3   Min.   :269.5  
    ##  1st Qu.:39.20    1st Qu.:279.2   1st Qu.:256.3   1st Qu.:406.6  
    ##  Median :52.00    Median :352.5   Median :270.0   Median :431.8  
    ##  Mean   :58.84    Mean   :351.7   Mean   :269.8   Mean   :433.5  
    ##  3rd Qu.:78.65    3rd Qu.:434.8   3rd Qu.:290.0   3rd Qu.:475.5  
    ##  Max.   :95.00    Max.   :549.0   Max.   :327.5   Max.   :496.3  
    ##      Width           Height      Minimal_Weight  Gross_Weight  Load_Capacity   
    ##  Min.   :164.5   Min.   :137.8   Min.   :1035   Min.   :1310   Min.   : 290.0  
    ##  1st Qu.:178.7   1st Qu.:151.2   1st Qu.:1516   1st Qu.:1882   1st Qu.: 440.0  
    ##  Median :180.2   Median :156.0   Median :1622   Median :2100   Median : 485.5  
    ##  Mean   :184.8   Mean   :155.0   Mean   :1821   Mean   :2268   Mean   : 510.5  
    ##  3rd Qu.:193.5   3rd Qu.:160.5   3rd Qu.:2249   3rd Qu.:2855   3rd Qu.: 565.0  
    ##  Max.   :255.8   Max.   :190.0   Max.   :2695   Max.   :3130   Max.   :1056.0  
    ##      Seats           Doors        Tire_Size       Max_Speed     Boot_Capacity  
    ##  Min.   :2.000   Min.   :3.00   Min.   :14.00   Min.   :130.0   Min.   :171.0  
    ##  1st Qu.:4.250   1st Qu.:5.00   1st Qu.:16.00   1st Qu.:146.2   1st Qu.:310.2  
    ##  Median :5.000   Median :5.00   Median :17.00   Median :160.0   Median :371.0  
    ##  Mean   :4.762   Mean   :4.81   Mean   :17.55   Mean   :169.5   Mean   :404.3  
    ##  3rd Qu.:5.000   3rd Qu.:5.00   3rd Qu.:19.00   3rd Qu.:187.5   3rd Qu.:497.0  
    ##  Max.   :8.000   Max.   :5.00   Max.   :21.00   Max.   :260.0   Max.   :660.0  
    ##  Acceleration_0-100 Max_DC_charging  Consumption   
    ##  Min.   : 2.800     Min.   : 22.0   Min.   :13.10  
    ##  1st Qu.: 6.800     1st Qu.: 62.5   1st Qu.:15.60  
    ##  Median : 7.900     Median :100.0   Median :16.88  
    ##  Mean   : 7.893     Mean   :109.7   Mean   :18.61  
    ##  3rd Qu.: 9.650     3rd Qu.:143.8   3rd Qu.:22.94  
    ##  Max.   :13.100     Max.   :270.0   Max.   :27.55

``` r
table(df_1$Brand)
```

    ## 
    ##          Audi           BMW       Citroën            DS         Honda 
    ##             6             3             1             1             2 
    ##       Hyundai        Jaguar           Kia         Mazda Mercedes-Benz 
    ##             3             1             4             1             1 
    ##          Mini        Nissan          Opel       Peugeot       Porsche 
    ##             1             2             2             1             4 
    ##       Renault         Skoda         Smart    Volkswagen 
    ##             2             1             2             4

``` r
table(df_1$`Type_Brakes`)
```

    ## 
    ##        disc (front + rear) disc (front) + drum (rear) 
    ##                         35                          7

``` r
table(df_1$`Drive_Type`)
```

    ## 
    ## 2WD (front)  2WD (rear)         4WD 
    ##          20          10          12

## 7.1 Exploratory Analysis

### 7.1.1 Mean of `Consumption` by categorical features

``` r
categorical_columns_1 <- c(2, 7:8)

df_categorical_1 <- df_1[, categorical_columns_1]

for (column in colnames(df_categorical_1)){
  
  if (column == 'Brand'){
    
    df_summarise <- df_1 %>%
      group_by(.data[[column]]) %>%
      summarise_at(vars(`Consumption`), list(consumption = mean)) %>% 
      arrange(desc(`consumption`), .by_group = TRUE) %>% top_n(10)
    
    plot <- ggplot(data = df_summarise, aes(x = reorder(.data[[column]], -consumption), y = consumption)) + 
      geom_text(aes(label = round(consumption, digits = 2)), position = position_stack(1.03), 
                color="black", size=3)+
      geom_bar(stat="identity",color="black",fill="steelblue") +
      labs(title = paste("Mean of Consumption by", column), x = column, y = "Mean of consumption (kWh/100km)")
    
    print(plot)
    
  }
  
  else {
    
    plot <- ggplot(df_1, aes(x = .data[[column]], y = `Consumption`)) + 
      geom_boxplot(color="black",fill="steelblue") +
      labs(title = paste("Mean of Consumption by", column), x = column, y = "Mean of consumption (kWh/100km)")
    
    print(plot)
  } 
  
}
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-2.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-21-3.png)<!-- -->

- Relation between `Consumption` and `Brand`

By this graph plot, we could categorize the top 10 `Brands` with the
**highest mean of `Consumption`** and verify that there are significant
difference among them. However, due to the reduced number of
observations in this dataframe, we should use this information with
caution.

- Relation between `Consumption` and `Type_Brakes`

The difference in the median of the two `Types_Brakes`, in terms of
`Consumption` **doesn’t seem to be significant**, even though the
difference in the pattern of the data (again, we have to consider that
we have more examples of one category).

- Relation between `Consumption` and `Drive_Type`

We can identify a clear possibility of the 4WD `Drive_Type` to be
**statistically different** from the other two categories, indicating
that this feature might be a good predictor for the model.

### 7.1.2 Mean of `Consumption` by numerical features

``` r
numerical_columns_1 <- c(4:6, 9:25)

df_numeric_1 <- df_1[, numerical_columns_1]

for (column in colnames(df_numeric_1)){
  
  if (column != "Consumption"){
    
    plot <- ggplot(df_numeric_1, aes(x = .data[[column]], y = `Consumption`)) +
      geom_point() + 
      labs(title = paste("Mean of Consumption x", column), x = column, y = "Mean of consumption (kWh/100km)")
    
    print(plot)
  }
}
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-1.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-2.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-3.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-4.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-5.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-6.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-7.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-8.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-9.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-10.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-11.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-12.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-13.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-14.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-15.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-16.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-17.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-18.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-22-19.png)<!-- -->

- Relation between `Consumption` and `Price`

In general, the **higher the `Price`, the higher the mean of
`Consumption`**. For smaller Prices though, there isn’t a clear
tendency. The relation between `Price` and the other numerical features
will be plot in the sequence.

- Relation between `Consumption` and `Power`

`Power` has a **positive correlation** with the mean of `Consumption`.

- Relation between `Consumption` and `Torque`

Similarly to `Power`, the graph plot indicates that **the higher the
`Torque`, the higher the `Consumption`**. In fact, `Power`and `Torque`
might have a high correlation between each other and this relation will
be examined later.

- Relation between `Consumption` and `Battery_Capacity`

It’s possible to see a **positive correlation between `Battery_Capacity`
and `Consumption`**.

- Relation between `Consumption` and `Range`

In this case, there isn’t a **clear tendency between the feature `Range`
and the car `Consumption`**.

- Relation between `Consumption`, `Wheelbase`, `Length`, `Width` and
  `Height`

It seems that, related to the car dimensions, **the higher that
dimension, the higher the car `Consumption` (except for the feature
`Height`)**.

- Relation between `Consumption`, `Minimal_Weight` and `Gross_Weight`

As expected, **higher values of `Weight` are associated to higher
`Consumption`**. Both graph plots have similar behavior, for this reason
the relation between these two features, along with the other
dimensions, will be examined later.

- Relation between `Consumption` and `Load_Capacity`

Once again, we observe a **positive correlation** between the
`Load_Capacity` of a car and its `Consumption`.

- Relation between `Consumption`, `Seats`, `Doors` and `Tire_Size`

Among these three features, only `Tire_Size` **might be a good
predictor** for `Consumption`.

- Relation between `Consumption`, `Max_Speed` and `Acceleration`

Both features seem to have **good positive correlation with
`Consumption`**. Nevertheless it’s important to consider a
multicollinearity between these two variables, since they represent
essencially the same result.

- Relation between `Consumption` and `Boot_Capacity`

Once again, it’s possible to affirm that **the higher the
`Boot_Capacity`, the higher the car `Consumption`**.

- Relation between `Consumption` and `Maximum_DC_Charging Power`

Although the feature has apperentaly **good positive correlation with
`Consumption`**, it’s important to extend our study on this variable,
specially comparing it to `Battery_Capacity` in order to avoid eventual
interference in the quality of our prediction model.

### 7.1.3 Relation between numerical columns and `Price`

``` r
for (column in colnames(df_numeric_1)){
  
  if (column != "Consumption" && column != "Price"){
    
    plot <- ggplot(df_numeric_1, aes(x = .data[[column]], y = `Price`)) +
      geom_point() + 
      labs(title = paste("Price x", column), x = column, y = "Price (PLN)")
    
    print(plot)
  }
}
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-1.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-2.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-3.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-4.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-5.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-6.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-7.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-8.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-9.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-10.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-11.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-12.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-13.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-14.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-15.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-16.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-17.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-23-18.png)<!-- -->

From the graph plots, it is possible to infer that the feature `Price`
has a high correlation with many of other variables in our dataset. In
fact, the price of a electrical car could be a sum of these other
features (among other factors). For this reason, we will decide **not to
consider `Price` as a predictor feature**.

### 7.1.4 Relation between `Power` and `Torque`

``` r
ggplot(df_numeric_1, aes(x = `Torque`, y = `Power`)) +
  geom_point() + 
  labs(title = "Power x Torque", x = "Torque (Nm)", y = "Power (kM)")
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

Both `Power` and `Torque` seem to be **highly correlated**. However,
they don’t indicate the same car characteristic: while the torque is a
measure of the car turning force, the engine power (or horsepower) is
responsible for the overall speed and power. We’ll decide later if it’s
interesting to keep both predictors.

### 7.1.5 Relation between the `Weight` of the vehicle and its dimensions

``` r
dimensions_car <- c("Wheelbase", "Length", "Width", "Height", "Gross_Weight")

for (feature in dimensions_car){
  
  plot <- ggplot(df_numeric_1, aes(x = .data[[feature]], y = `Minimal_Weight`)) +
    geom_point() +
    labs(title = paste("Minimal Empty Weight x", feature), x = feature, y = "Empty Weight (kg)")
  
  print(plot)
}
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-25-2.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-25-3.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-25-4.png)<!-- -->![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-25-5.png)<!-- -->

As we previously supposed, the features seem to have **high correlation
between each other**, especially `Empty_Weight` and `Gross_Weight`. It
is plausible to believe, in fact, that a car dimensions constitute great
part of it’s weight. If we follow with all of the features, it is
possible for our regression model to be **negatively affected**.

### 7.1.6 Relation between `Battery_Capacity` and `Maximum_DC_ Charging_ Power`

``` r
ggplot(df_numeric_1, aes(x = `Max_DC_charging`, y = `Battery_Capacity`)) +
  geom_point() + 
  labs(title = "Battery Capacity x Maximum DC charging Power",
       x = "DC charging power (kw)", y = "Battery Capacity (kWh)")
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-26-1.png)<!-- -->

It is believed that both, `Battery_Capacity` and
`Maximum_DC_Charging_ Power`, are similar features and are **postive
correlated**. For this reason, we decided to not follow with one of
them.

### 7.1.7 Histogram + Boxplot of `Consumption`

``` r
plot1 <- ggplot(df_1, aes(x = Consumption), binwidth = 30) +
  geom_histogram(alpha = 1, bins=30, color="black",fill="steelblue") +
  labs(title = "Histogram of Consumption", x = "Mean of consumption (kWh/100km)", y = "Count")

plot2 <- ggplot(df_1, aes(x = Consumption)) +
  geom_boxplot(color="black",fill="steelblue") + 
  labs(title = "Boxplot of Consumption", x = "Mean of consumption (kWh/100km)", y = "")

grid.arrange(plot1, plot2, ncol = 1)
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-27-1.png)<!-- -->

The histogram of `Consumption` indicates that this features **does not
follow a normal distribution**. The bulk of data are concentrated on
**smaller values of `Consumption`**, even though both, histogram and
boxplot, shows a distortion towards highest values.

## 7.2 Feature correlation

We can now visualize a matrix containing the correlation among all
features. When the level of significance is less than 5% we chose to
omit the corresponding correlation.

``` r
matrix_correlation <- cor(df_numeric_1)

cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

p.mat <- cor.mtest(df_numeric_1)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(matrix_correlation, method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black",
         tl.col="black", tl.srt=90, 
         p.mat = p.mat, sig.level = 0.05, insig = "blank", 
         diag=TRUE)
```

![](electric_cars_prediction_files/figure-gfm/fig1-1.png)<!-- -->

## 7.3 Statistical tests for categorical features

In the case of categorical features, `Type_ Brakes`, `Drive_Type` and
`Brand`, it is important to check their statistical significance related
to our target feature `Consumption`, in order to evaluate their usage in
our prediction model. We’ll start by verifying the normality
distribution of the target feature with Shapiro-Wilk test:

``` r
shapiro_test <- shapiro.test(df_1$Consumption)
shapiro_test
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  df_1$Consumption
    ## W = 0.86663, p-value = 0.0001665

Assuming a 5% significance level, the p-value indicates that we can
**reject the hypothesis of normality**. Because of this, we won’t be
able to use parametric tests for our categorical variables. In the next
verifications we made use of Kruskal-Wallis test:

``` r
kruskal_test_1 <- kruskal.test(Consumption ~ Type_Brakes, data = df_1)
kruskal_test_1
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Type_Brakes
    ## Kruskal-Wallis chi-squared = 2.7919, df = 1, p-value = 0.09474

Since the p-value is greater than 5%, we **failed to reject the null
hypothesis**, that is, the medians of the each type of brake are similar
(actually, we had checked this behavior before in the graph plots).

``` r
kruskal_test_2 <- kruskal.test(Consumption ~ Drive_Type, data = df_1)
kruskal_test_2
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Drive_Type
    ## Kruskal-Wallis chi-squared = 22.57, df = 2, p-value = 1.256e-05

For the feature `Drive_Type`, **at least one category has the median
statistically different from the others**.

``` r
kruskal_test_3 <- kruskal.test(Consumption ~ Brand, data = df_1)
kruskal_test_3
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Brand
    ## Kruskal-Wallis chi-squared = 34.402, df = 18, p-value = 0.01123

According to our criteria, the feature `Brand` could be also
**statistically significant**. In order to evaluate two paired groups,
we’ll make use of the non parametric Wilcoxon test.

``` r
wilcoxon_test_2 <- pairwise.wilcox.test(df_1$Consumption,
                                      df_1$Drive_Type,
                                      p.adjust.method="bonferroni", exact = FALSE)
wilcoxon_test_2
```

    ## 
    ##  Pairwise comparisons using Wilcoxon rank sum test with continuity correction 
    ## 
    ## data:  df_1$Consumption and df_1$Drive_Type 
    ## 
    ##            2WD (front) 2WD (rear)
    ## 2WD (rear) 1.00000     -         
    ## 4WD        6e-05       0.00026   
    ## 
    ## P value adjustment method: bonferroni

Two categories, `2WD (front)` and `2WD (rear)`, are **not statistically
different from each other**, but when compared to `4WD` **both passed
our p-value test**. We decided to follow with `4WD` and `2WD (front)`as
input to our model (these two categories presented the lowest p-value)
and let `2WD (rear)` as a baseline level.

``` r
wilcoxon_test_3 <- pairwise.wilcox.test(df_1$Consumption,
                                      df_1$Brand,
                                      p.adjust.method="bonferroni", exact = FALSE)
wilcoxon_test_3
```

    ## 
    ##  Pairwise comparisons using Wilcoxon rank sum test with continuity correction 
    ## 
    ## data:  df_1$Consumption and df_1$Brand 
    ## 
    ##               Audi BMW Citroën DS Honda Hyundai Jaguar Kia Mazda Mercedes-Benz
    ## BMW           1    -   -       -  -     -       -      -   -     -            
    ## Citroën       1    1   -       -  -     -       -      -   -     -            
    ## DS            1    1   1       -  -     -       -      -   -     -            
    ## Honda         1    1   1       1  -     -       -      -   -     -            
    ## Hyundai       1    1   1       1  1     -       -      -   -     -            
    ## Jaguar        1    1   1       1  1     1       -      -   -     -            
    ## Kia           1    1   1       1  1     1       1      -   -     -            
    ## Mazda         1    1   1       1  1     1       1      1   -     -            
    ## Mercedes-Benz 1    1   1       1  1     1       1      1   1     -            
    ## Mini          1    1   1       1  1     1       1      1   1     1            
    ## Nissan        1    1   1       1  1     1       1      1   1     1            
    ## Opel          1    1   1       1  1     1       1      1   1     1            
    ## Peugeot       1    1   1       1  1     1       1      1   1     1            
    ## Porsche       1    1   1       1  1     1       1      1   1     1            
    ## Renault       1    1   1       1  1     1       1      1   1     1            
    ## Skoda         1    1   1       1  1     1       1      1   1     1            
    ## Smart         1    1   1       1  1     1       1      1   1     1            
    ## Volkswagen    1    1   1       1  1     1       1      1   1     1            
    ##               Mini Nissan Opel Peugeot Porsche Renault Skoda Smart
    ## BMW           -    -      -    -       -       -       -     -    
    ## Citroën       -    -      -    -       -       -       -     -    
    ## DS            -    -      -    -       -       -       -     -    
    ## Honda         -    -      -    -       -       -       -     -    
    ## Hyundai       -    -      -    -       -       -       -     -    
    ## Jaguar        -    -      -    -       -       -       -     -    
    ## Kia           -    -      -    -       -       -       -     -    
    ## Mazda         -    -      -    -       -       -       -     -    
    ## Mercedes-Benz -    -      -    -       -       -       -     -    
    ## Mini          -    -      -    -       -       -       -     -    
    ## Nissan        1    -      -    -       -       -       -     -    
    ## Opel          1    1      -    -       -       -       -     -    
    ## Peugeot       1    1      1    -       -       -       -     -    
    ## Porsche       1    1      1    1       -       -       -     -    
    ## Renault       1    1      1    1       1       -       -     -    
    ## Skoda         1    1      1    1       1       1       -     -    
    ## Smart         1    1      1    1       1       1       1     -    
    ## Volkswagen    1    1      1    1       1       1       1     1    
    ## 
    ## P value adjustment method: bonferroni

**Any of the pairs evaluated seem to have statistical significance**. In
fact, the `Brand` feature shouldn’t be consider a good predictor in the
context of this project.

## 7.4 Splitting data

From our exploratory analysis, we decided to choose the following
features as input to our regression prediction model: `Power`, `Torque`,
`Drive_Type`, `Battery_Capacity`, `Gross_Weight`, `Load_Capacity`,
`Tire_Size`, `Max_Speed` and `Boot_Capacity`. As we previously
concluded, the other features either have multicollinearity or were
already represented by the chosen ones.

``` r
model_1 <- df_1[, c(5, 6, 8, 9, 16, 17, 20:22, 25)]
```

We will now convert our categorical feature `Drive_Type` into Dummy
variables:

``` r
model_1 <- dummy_cols(model_1, select_columns = 'Drive_Type')
```

We can now exclude `Drive_Type` and `Drive_Type_2WD (rear)`:

``` r
model_1 <- model_1[, -c(3, 12)]
```

``` r
colnames(model_1)[10] <- "Drive_Type_2WD"
```

Our final dataframe contains a smaller number of observations (we
omitted all NA values). For this reason, we splitted 20% of it for our
final test set.

``` r
set.seed(57)

sample <- sample.split(model_1$Consumption, SplitRatio = 0.8)
train_validation  <- subset(model_1, sample == TRUE)
test   <- subset(model_1, sample == FALSE)

dim(model_1)
```

    ## [1] 42 11

``` r
dim(train_validation)
```

    ## [1] 33 11

``` r
dim(test)
```

    ## [1]  9 11

## 7.5 Standardizing data

Before the process of training we need to perform stardardization of the
features, so that we **prevent features with wider ranges from
dominating others**. For this purpose, we’ll take the mean and standard
deviation from the training set and use them to stardardize both, the
training and test set.

``` r
train_means <- data.frame(as.list(train_validation %>% apply(2, mean)))

train_std <- data.frame(as.list(train_validation %>% apply(2, sd)))

col_names <- names(train_validation[, -9])

for (i in col_names){
  train_validation[, i] <- (train_validation[, i] - train_means[, i])/train_std[, i]
  test[, i] <-  (test[, i] - train_means[, i])/train_std[, i]
}
```

## 7.6 Choosing and training models

For our regression project, we decided to test the following machine
learning models: **Linear Regression**, **Ridge Regression**, **Random
Forest** and **XGBoost**.

For our first running, we will consider all the previously selected
features, which will trained across a 5-fold-cross validation method (in
order to avoid randomness of evaluation).

``` r
models = c("lm", "ridge", "rf", "xgbDART")

model_trained_1 <- c()

set.seed(57) 

train.control <- trainControl(method = "cv", number = 5)

for (model in models){
  model_trained_1[[model]] <- train(`Consumption` ~ ., data = train_validation, 
                                    method = model,
                                    trControl = train.control,
                                    metric = 'Rsquared', 
                                    verbosity = 0)
}

print(model_trained_1)
```

    ## $lm
    ## Linear Regression 
    ## 
    ## 33 samples
    ## 10 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 27, 27, 25, 27, 26 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE     
    ##   1.503284  0.9227503  1.285904
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE
    ## 
    ## $ridge
    ## Ridge Regression 
    ## 
    ## 33 samples
    ## 10 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 27, 25, 27, 26, 27 
    ## Resampling results across tuning parameters:
    ## 
    ##   lambda  RMSE      Rsquared   MAE     
    ##   0e+00   1.385231  0.8894310  1.171725
    ##   1e-04   1.385712  0.8889953  1.172250
    ##   1e-01   1.677046  0.8179649  1.314083
    ## 
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final value used for the model was lambda = 0.
    ## 
    ## $rf
    ## Random Forest 
    ## 
    ## 33 samples
    ## 10 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 27, 25, 28, 26, 26 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared   MAE     
    ##    2    1.567371  0.8390737  1.106374
    ##    6    1.532912  0.8335373  1.084861
    ##   10    1.522997  0.8336572  1.110569
    ## 
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 2.
    ## 
    ## $xgbDART
    ## eXtreme Gradient Boosting 
    ## 
    ## 33 samples
    ## 10 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 28, 25, 27, 27, 25 
    ## Resampling results across tuning parameters:
    ## 
    ##   max_depth  eta  rate_drop  skip_drop  subsample  colsample_bytree  nrounds
    ##   1          0.3  0.01       0.05       0.50       0.6                50    
    ##   1          0.3  0.01       0.05       0.50       0.6               100    
    ##   1          0.3  0.01       0.05       0.50       0.6               150    
    ##   1          0.3  0.01       0.05       0.50       0.8                50    
    ##   1          0.3  0.01       0.05       0.50       0.8               100    
    ##   1          0.3  0.01       0.05       0.50       0.8               150    
    ##   1          0.3  0.01       0.05       0.75       0.6                50    
    ##   1          0.3  0.01       0.05       0.75       0.6               100    
    ##   1          0.3  0.01       0.05       0.75       0.6               150    
    ##   1          0.3  0.01       0.05       0.75       0.8                50    
    ##   1          0.3  0.01       0.05       0.75       0.8               100    
    ##   1          0.3  0.01       0.05       0.75       0.8               150    
    ##   1          0.3  0.01       0.05       1.00       0.6                50    
    ##   1          0.3  0.01       0.05       1.00       0.6               100    
    ##   1          0.3  0.01       0.05       1.00       0.6               150    
    ##   1          0.3  0.01       0.05       1.00       0.8                50    
    ##   1          0.3  0.01       0.05       1.00       0.8               100    
    ##   1          0.3  0.01       0.05       1.00       0.8               150    
    ##   1          0.3  0.01       0.95       0.50       0.6                50    
    ##   1          0.3  0.01       0.95       0.50       0.6               100    
    ##   1          0.3  0.01       0.95       0.50       0.6               150    
    ##   1          0.3  0.01       0.95       0.50       0.8                50    
    ##   1          0.3  0.01       0.95       0.50       0.8               100    
    ##   1          0.3  0.01       0.95       0.50       0.8               150    
    ##   1          0.3  0.01       0.95       0.75       0.6                50    
    ##   1          0.3  0.01       0.95       0.75       0.6               100    
    ##   1          0.3  0.01       0.95       0.75       0.6               150    
    ##   1          0.3  0.01       0.95       0.75       0.8                50    
    ##   1          0.3  0.01       0.95       0.75       0.8               100    
    ##   1          0.3  0.01       0.95       0.75       0.8               150    
    ##   1          0.3  0.01       0.95       1.00       0.6                50    
    ##   1          0.3  0.01       0.95       1.00       0.6               100    
    ##   1          0.3  0.01       0.95       1.00       0.6               150    
    ##   1          0.3  0.01       0.95       1.00       0.8                50    
    ##   1          0.3  0.01       0.95       1.00       0.8               100    
    ##   1          0.3  0.01       0.95       1.00       0.8               150    
    ##   1          0.3  0.50       0.05       0.50       0.6                50    
    ##   1          0.3  0.50       0.05       0.50       0.6               100    
    ##   1          0.3  0.50       0.05       0.50       0.6               150    
    ##   1          0.3  0.50       0.05       0.50       0.8                50    
    ##   1          0.3  0.50       0.05       0.50       0.8               100    
    ##   1          0.3  0.50       0.05       0.50       0.8               150    
    ##   1          0.3  0.50       0.05       0.75       0.6                50    
    ##   1          0.3  0.50       0.05       0.75       0.6               100    
    ##   1          0.3  0.50       0.05       0.75       0.6               150    
    ##   1          0.3  0.50       0.05       0.75       0.8                50    
    ##   1          0.3  0.50       0.05       0.75       0.8               100    
    ##   1          0.3  0.50       0.05       0.75       0.8               150    
    ##   1          0.3  0.50       0.05       1.00       0.6                50    
    ##   1          0.3  0.50       0.05       1.00       0.6               100    
    ##   1          0.3  0.50       0.05       1.00       0.6               150    
    ##   1          0.3  0.50       0.05       1.00       0.8                50    
    ##   1          0.3  0.50       0.05       1.00       0.8               100    
    ##   1          0.3  0.50       0.05       1.00       0.8               150    
    ##   1          0.3  0.50       0.95       0.50       0.6                50    
    ##   1          0.3  0.50       0.95       0.50       0.6               100    
    ##   1          0.3  0.50       0.95       0.50       0.6               150    
    ##   1          0.3  0.50       0.95       0.50       0.8                50    
    ##   1          0.3  0.50       0.95       0.50       0.8               100    
    ##   1          0.3  0.50       0.95       0.50       0.8               150    
    ##   1          0.3  0.50       0.95       0.75       0.6                50    
    ##   1          0.3  0.50       0.95       0.75       0.6               100    
    ##   1          0.3  0.50       0.95       0.75       0.6               150    
    ##   1          0.3  0.50       0.95       0.75       0.8                50    
    ##   1          0.3  0.50       0.95       0.75       0.8               100    
    ##   1          0.3  0.50       0.95       0.75       0.8               150    
    ##   1          0.3  0.50       0.95       1.00       0.6                50    
    ##   1          0.3  0.50       0.95       1.00       0.6               100    
    ##   1          0.3  0.50       0.95       1.00       0.6               150    
    ##   1          0.3  0.50       0.95       1.00       0.8                50    
    ##   1          0.3  0.50       0.95       1.00       0.8               100    
    ##   1          0.3  0.50       0.95       1.00       0.8               150    
    ##   1          0.4  0.01       0.05       0.50       0.6                50    
    ##   1          0.4  0.01       0.05       0.50       0.6               100    
    ##   1          0.4  0.01       0.05       0.50       0.6               150    
    ##   1          0.4  0.01       0.05       0.50       0.8                50    
    ##   1          0.4  0.01       0.05       0.50       0.8               100    
    ##   1          0.4  0.01       0.05       0.50       0.8               150    
    ##   1          0.4  0.01       0.05       0.75       0.6                50    
    ##   1          0.4  0.01       0.05       0.75       0.6               100    
    ##   1          0.4  0.01       0.05       0.75       0.6               150    
    ##   1          0.4  0.01       0.05       0.75       0.8                50    
    ##   1          0.4  0.01       0.05       0.75       0.8               100    
    ##   1          0.4  0.01       0.05       0.75       0.8               150    
    ##   1          0.4  0.01       0.05       1.00       0.6                50    
    ##   1          0.4  0.01       0.05       1.00       0.6               100    
    ##   1          0.4  0.01       0.05       1.00       0.6               150    
    ##   1          0.4  0.01       0.05       1.00       0.8                50    
    ##   1          0.4  0.01       0.05       1.00       0.8               100    
    ##   1          0.4  0.01       0.05       1.00       0.8               150    
    ##   1          0.4  0.01       0.95       0.50       0.6                50    
    ##   1          0.4  0.01       0.95       0.50       0.6               100    
    ##   1          0.4  0.01       0.95       0.50       0.6               150    
    ##   1          0.4  0.01       0.95       0.50       0.8                50    
    ##   1          0.4  0.01       0.95       0.50       0.8               100    
    ##   1          0.4  0.01       0.95       0.50       0.8               150    
    ##   1          0.4  0.01       0.95       0.75       0.6                50    
    ##   1          0.4  0.01       0.95       0.75       0.6               100    
    ##   1          0.4  0.01       0.95       0.75       0.6               150    
    ##   1          0.4  0.01       0.95       0.75       0.8                50    
    ##   1          0.4  0.01       0.95       0.75       0.8               100    
    ##   1          0.4  0.01       0.95       0.75       0.8               150    
    ##   1          0.4  0.01       0.95       1.00       0.6                50    
    ##   1          0.4  0.01       0.95       1.00       0.6               100    
    ##   1          0.4  0.01       0.95       1.00       0.6               150    
    ##   1          0.4  0.01       0.95       1.00       0.8                50    
    ##   1          0.4  0.01       0.95       1.00       0.8               100    
    ##   1          0.4  0.01       0.95       1.00       0.8               150    
    ##   1          0.4  0.50       0.05       0.50       0.6                50    
    ##   1          0.4  0.50       0.05       0.50       0.6               100    
    ##   1          0.4  0.50       0.05       0.50       0.6               150    
    ##   1          0.4  0.50       0.05       0.50       0.8                50    
    ##   1          0.4  0.50       0.05       0.50       0.8               100    
    ##   1          0.4  0.50       0.05       0.50       0.8               150    
    ##   1          0.4  0.50       0.05       0.75       0.6                50    
    ##   1          0.4  0.50       0.05       0.75       0.6               100    
    ##   1          0.4  0.50       0.05       0.75       0.6               150    
    ##   1          0.4  0.50       0.05       0.75       0.8                50    
    ##   1          0.4  0.50       0.05       0.75       0.8               100    
    ##   1          0.4  0.50       0.05       0.75       0.8               150    
    ##   1          0.4  0.50       0.05       1.00       0.6                50    
    ##   1          0.4  0.50       0.05       1.00       0.6               100    
    ##   1          0.4  0.50       0.05       1.00       0.6               150    
    ##   1          0.4  0.50       0.05       1.00       0.8                50    
    ##   1          0.4  0.50       0.05       1.00       0.8               100    
    ##   1          0.4  0.50       0.05       1.00       0.8               150    
    ##   1          0.4  0.50       0.95       0.50       0.6                50    
    ##   1          0.4  0.50       0.95       0.50       0.6               100    
    ##   1          0.4  0.50       0.95       0.50       0.6               150    
    ##   1          0.4  0.50       0.95       0.50       0.8                50    
    ##   1          0.4  0.50       0.95       0.50       0.8               100    
    ##   1          0.4  0.50       0.95       0.50       0.8               150    
    ##   1          0.4  0.50       0.95       0.75       0.6                50    
    ##   1          0.4  0.50       0.95       0.75       0.6               100    
    ##   1          0.4  0.50       0.95       0.75       0.6               150    
    ##   1          0.4  0.50       0.95       0.75       0.8                50    
    ##   1          0.4  0.50       0.95       0.75       0.8               100    
    ##   1          0.4  0.50       0.95       0.75       0.8               150    
    ##   1          0.4  0.50       0.95       1.00       0.6                50    
    ##   1          0.4  0.50       0.95       1.00       0.6               100    
    ##   1          0.4  0.50       0.95       1.00       0.6               150    
    ##   1          0.4  0.50       0.95       1.00       0.8                50    
    ##   1          0.4  0.50       0.95       1.00       0.8               100    
    ##   1          0.4  0.50       0.95       1.00       0.8               150    
    ##   2          0.3  0.01       0.05       0.50       0.6                50    
    ##   2          0.3  0.01       0.05       0.50       0.6               100    
    ##   2          0.3  0.01       0.05       0.50       0.6               150    
    ##   2          0.3  0.01       0.05       0.50       0.8                50    
    ##   2          0.3  0.01       0.05       0.50       0.8               100    
    ##   2          0.3  0.01       0.05       0.50       0.8               150    
    ##   2          0.3  0.01       0.05       0.75       0.6                50    
    ##   2          0.3  0.01       0.05       0.75       0.6               100    
    ##   2          0.3  0.01       0.05       0.75       0.6               150    
    ##   2          0.3  0.01       0.05       0.75       0.8                50    
    ##   2          0.3  0.01       0.05       0.75       0.8               100    
    ##   2          0.3  0.01       0.05       0.75       0.8               150    
    ##   2          0.3  0.01       0.05       1.00       0.6                50    
    ##   2          0.3  0.01       0.05       1.00       0.6               100    
    ##   2          0.3  0.01       0.05       1.00       0.6               150    
    ##   2          0.3  0.01       0.05       1.00       0.8                50    
    ##   2          0.3  0.01       0.05       1.00       0.8               100    
    ##   2          0.3  0.01       0.05       1.00       0.8               150    
    ##   2          0.3  0.01       0.95       0.50       0.6                50    
    ##   2          0.3  0.01       0.95       0.50       0.6               100    
    ##   2          0.3  0.01       0.95       0.50       0.6               150    
    ##   2          0.3  0.01       0.95       0.50       0.8                50    
    ##   2          0.3  0.01       0.95       0.50       0.8               100    
    ##   2          0.3  0.01       0.95       0.50       0.8               150    
    ##   2          0.3  0.01       0.95       0.75       0.6                50    
    ##   2          0.3  0.01       0.95       0.75       0.6               100    
    ##   2          0.3  0.01       0.95       0.75       0.6               150    
    ##   2          0.3  0.01       0.95       0.75       0.8                50    
    ##   2          0.3  0.01       0.95       0.75       0.8               100    
    ##   2          0.3  0.01       0.95       0.75       0.8               150    
    ##   2          0.3  0.01       0.95       1.00       0.6                50    
    ##   2          0.3  0.01       0.95       1.00       0.6               100    
    ##   2          0.3  0.01       0.95       1.00       0.6               150    
    ##   2          0.3  0.01       0.95       1.00       0.8                50    
    ##   2          0.3  0.01       0.95       1.00       0.8               100    
    ##   2          0.3  0.01       0.95       1.00       0.8               150    
    ##   2          0.3  0.50       0.05       0.50       0.6                50    
    ##   2          0.3  0.50       0.05       0.50       0.6               100    
    ##   2          0.3  0.50       0.05       0.50       0.6               150    
    ##   2          0.3  0.50       0.05       0.50       0.8                50    
    ##   2          0.3  0.50       0.05       0.50       0.8               100    
    ##   2          0.3  0.50       0.05       0.50       0.8               150    
    ##   2          0.3  0.50       0.05       0.75       0.6                50    
    ##   2          0.3  0.50       0.05       0.75       0.6               100    
    ##   2          0.3  0.50       0.05       0.75       0.6               150    
    ##   2          0.3  0.50       0.05       0.75       0.8                50    
    ##   2          0.3  0.50       0.05       0.75       0.8               100    
    ##   2          0.3  0.50       0.05       0.75       0.8               150    
    ##   2          0.3  0.50       0.05       1.00       0.6                50    
    ##   2          0.3  0.50       0.05       1.00       0.6               100    
    ##   2          0.3  0.50       0.05       1.00       0.6               150    
    ##   2          0.3  0.50       0.05       1.00       0.8                50    
    ##   2          0.3  0.50       0.05       1.00       0.8               100    
    ##   2          0.3  0.50       0.05       1.00       0.8               150    
    ##   2          0.3  0.50       0.95       0.50       0.6                50    
    ##   2          0.3  0.50       0.95       0.50       0.6               100    
    ##   2          0.3  0.50       0.95       0.50       0.6               150    
    ##   2          0.3  0.50       0.95       0.50       0.8                50    
    ##   2          0.3  0.50       0.95       0.50       0.8               100    
    ##   2          0.3  0.50       0.95       0.50       0.8               150    
    ##   2          0.3  0.50       0.95       0.75       0.6                50    
    ##   2          0.3  0.50       0.95       0.75       0.6               100    
    ##   2          0.3  0.50       0.95       0.75       0.6               150    
    ##   2          0.3  0.50       0.95       0.75       0.8                50    
    ##   2          0.3  0.50       0.95       0.75       0.8               100    
    ##   2          0.3  0.50       0.95       0.75       0.8               150    
    ##   2          0.3  0.50       0.95       1.00       0.6                50    
    ##   2          0.3  0.50       0.95       1.00       0.6               100    
    ##   2          0.3  0.50       0.95       1.00       0.6               150    
    ##   2          0.3  0.50       0.95       1.00       0.8                50    
    ##   2          0.3  0.50       0.95       1.00       0.8               100    
    ##   2          0.3  0.50       0.95       1.00       0.8               150    
    ##   2          0.4  0.01       0.05       0.50       0.6                50    
    ##   2          0.4  0.01       0.05       0.50       0.6               100    
    ##   2          0.4  0.01       0.05       0.50       0.6               150    
    ##   2          0.4  0.01       0.05       0.50       0.8                50    
    ##   2          0.4  0.01       0.05       0.50       0.8               100    
    ##   2          0.4  0.01       0.05       0.50       0.8               150    
    ##   2          0.4  0.01       0.05       0.75       0.6                50    
    ##   2          0.4  0.01       0.05       0.75       0.6               100    
    ##   2          0.4  0.01       0.05       0.75       0.6               150    
    ##   2          0.4  0.01       0.05       0.75       0.8                50    
    ##   2          0.4  0.01       0.05       0.75       0.8               100    
    ##   2          0.4  0.01       0.05       0.75       0.8               150    
    ##   2          0.4  0.01       0.05       1.00       0.6                50    
    ##   2          0.4  0.01       0.05       1.00       0.6               100    
    ##   2          0.4  0.01       0.05       1.00       0.6               150    
    ##   2          0.4  0.01       0.05       1.00       0.8                50    
    ##   2          0.4  0.01       0.05       1.00       0.8               100    
    ##   2          0.4  0.01       0.05       1.00       0.8               150    
    ##   2          0.4  0.01       0.95       0.50       0.6                50    
    ##   2          0.4  0.01       0.95       0.50       0.6               100    
    ##   2          0.4  0.01       0.95       0.50       0.6               150    
    ##   2          0.4  0.01       0.95       0.50       0.8                50    
    ##   2          0.4  0.01       0.95       0.50       0.8               100    
    ##   2          0.4  0.01       0.95       0.50       0.8               150    
    ##   2          0.4  0.01       0.95       0.75       0.6                50    
    ##   2          0.4  0.01       0.95       0.75       0.6               100    
    ##   2          0.4  0.01       0.95       0.75       0.6               150    
    ##   2          0.4  0.01       0.95       0.75       0.8                50    
    ##   2          0.4  0.01       0.95       0.75       0.8               100    
    ##   2          0.4  0.01       0.95       0.75       0.8               150    
    ##   2          0.4  0.01       0.95       1.00       0.6                50    
    ##   2          0.4  0.01       0.95       1.00       0.6               100    
    ##   2          0.4  0.01       0.95       1.00       0.6               150    
    ##   2          0.4  0.01       0.95       1.00       0.8                50    
    ##   2          0.4  0.01       0.95       1.00       0.8               100    
    ##   2          0.4  0.01       0.95       1.00       0.8               150    
    ##   2          0.4  0.50       0.05       0.50       0.6                50    
    ##   2          0.4  0.50       0.05       0.50       0.6               100    
    ##   2          0.4  0.50       0.05       0.50       0.6               150    
    ##   2          0.4  0.50       0.05       0.50       0.8                50    
    ##   2          0.4  0.50       0.05       0.50       0.8               100    
    ##   2          0.4  0.50       0.05       0.50       0.8               150    
    ##   2          0.4  0.50       0.05       0.75       0.6                50    
    ##   2          0.4  0.50       0.05       0.75       0.6               100    
    ##   2          0.4  0.50       0.05       0.75       0.6               150    
    ##   2          0.4  0.50       0.05       0.75       0.8                50    
    ##   2          0.4  0.50       0.05       0.75       0.8               100    
    ##   2          0.4  0.50       0.05       0.75       0.8               150    
    ##   2          0.4  0.50       0.05       1.00       0.6                50    
    ##   2          0.4  0.50       0.05       1.00       0.6               100    
    ##   2          0.4  0.50       0.05       1.00       0.6               150    
    ##   2          0.4  0.50       0.05       1.00       0.8                50    
    ##   2          0.4  0.50       0.05       1.00       0.8               100    
    ##   2          0.4  0.50       0.05       1.00       0.8               150    
    ##   2          0.4  0.50       0.95       0.50       0.6                50    
    ##   2          0.4  0.50       0.95       0.50       0.6               100    
    ##   2          0.4  0.50       0.95       0.50       0.6               150    
    ##   2          0.4  0.50       0.95       0.50       0.8                50    
    ##   2          0.4  0.50       0.95       0.50       0.8               100    
    ##   2          0.4  0.50       0.95       0.50       0.8               150    
    ##   2          0.4  0.50       0.95       0.75       0.6                50    
    ##   2          0.4  0.50       0.95       0.75       0.6               100    
    ##   2          0.4  0.50       0.95       0.75       0.6               150    
    ##   2          0.4  0.50       0.95       0.75       0.8                50    
    ##   2          0.4  0.50       0.95       0.75       0.8               100    
    ##   2          0.4  0.50       0.95       0.75       0.8               150    
    ##   2          0.4  0.50       0.95       1.00       0.6                50    
    ##   2          0.4  0.50       0.95       1.00       0.6               100    
    ##   2          0.4  0.50       0.95       1.00       0.6               150    
    ##   2          0.4  0.50       0.95       1.00       0.8                50    
    ##   2          0.4  0.50       0.95       1.00       0.8               100    
    ##   2          0.4  0.50       0.95       1.00       0.8               150    
    ##   3          0.3  0.01       0.05       0.50       0.6                50    
    ##   3          0.3  0.01       0.05       0.50       0.6               100    
    ##   3          0.3  0.01       0.05       0.50       0.6               150    
    ##   3          0.3  0.01       0.05       0.50       0.8                50    
    ##   3          0.3  0.01       0.05       0.50       0.8               100    
    ##   3          0.3  0.01       0.05       0.50       0.8               150    
    ##   3          0.3  0.01       0.05       0.75       0.6                50    
    ##   3          0.3  0.01       0.05       0.75       0.6               100    
    ##   3          0.3  0.01       0.05       0.75       0.6               150    
    ##   3          0.3  0.01       0.05       0.75       0.8                50    
    ##   3          0.3  0.01       0.05       0.75       0.8               100    
    ##   3          0.3  0.01       0.05       0.75       0.8               150    
    ##   3          0.3  0.01       0.05       1.00       0.6                50    
    ##   3          0.3  0.01       0.05       1.00       0.6               100    
    ##   3          0.3  0.01       0.05       1.00       0.6               150    
    ##   3          0.3  0.01       0.05       1.00       0.8                50    
    ##   3          0.3  0.01       0.05       1.00       0.8               100    
    ##   3          0.3  0.01       0.05       1.00       0.8               150    
    ##   3          0.3  0.01       0.95       0.50       0.6                50    
    ##   3          0.3  0.01       0.95       0.50       0.6               100    
    ##   3          0.3  0.01       0.95       0.50       0.6               150    
    ##   3          0.3  0.01       0.95       0.50       0.8                50    
    ##   3          0.3  0.01       0.95       0.50       0.8               100    
    ##   3          0.3  0.01       0.95       0.50       0.8               150    
    ##   3          0.3  0.01       0.95       0.75       0.6                50    
    ##   3          0.3  0.01       0.95       0.75       0.6               100    
    ##   3          0.3  0.01       0.95       0.75       0.6               150    
    ##   3          0.3  0.01       0.95       0.75       0.8                50    
    ##   3          0.3  0.01       0.95       0.75       0.8               100    
    ##   3          0.3  0.01       0.95       0.75       0.8               150    
    ##   3          0.3  0.01       0.95       1.00       0.6                50    
    ##   3          0.3  0.01       0.95       1.00       0.6               100    
    ##   3          0.3  0.01       0.95       1.00       0.6               150    
    ##   3          0.3  0.01       0.95       1.00       0.8                50    
    ##   3          0.3  0.01       0.95       1.00       0.8               100    
    ##   3          0.3  0.01       0.95       1.00       0.8               150    
    ##   3          0.3  0.50       0.05       0.50       0.6                50    
    ##   3          0.3  0.50       0.05       0.50       0.6               100    
    ##   3          0.3  0.50       0.05       0.50       0.6               150    
    ##   3          0.3  0.50       0.05       0.50       0.8                50    
    ##   3          0.3  0.50       0.05       0.50       0.8               100    
    ##   3          0.3  0.50       0.05       0.50       0.8               150    
    ##   3          0.3  0.50       0.05       0.75       0.6                50    
    ##   3          0.3  0.50       0.05       0.75       0.6               100    
    ##   3          0.3  0.50       0.05       0.75       0.6               150    
    ##   3          0.3  0.50       0.05       0.75       0.8                50    
    ##   3          0.3  0.50       0.05       0.75       0.8               100    
    ##   3          0.3  0.50       0.05       0.75       0.8               150    
    ##   3          0.3  0.50       0.05       1.00       0.6                50    
    ##   3          0.3  0.50       0.05       1.00       0.6               100    
    ##   3          0.3  0.50       0.05       1.00       0.6               150    
    ##   3          0.3  0.50       0.05       1.00       0.8                50    
    ##   3          0.3  0.50       0.05       1.00       0.8               100    
    ##   3          0.3  0.50       0.05       1.00       0.8               150    
    ##   3          0.3  0.50       0.95       0.50       0.6                50    
    ##   3          0.3  0.50       0.95       0.50       0.6               100    
    ##   3          0.3  0.50       0.95       0.50       0.6               150    
    ##   3          0.3  0.50       0.95       0.50       0.8                50    
    ##   3          0.3  0.50       0.95       0.50       0.8               100    
    ##   3          0.3  0.50       0.95       0.50       0.8               150    
    ##   3          0.3  0.50       0.95       0.75       0.6                50    
    ##   3          0.3  0.50       0.95       0.75       0.6               100    
    ##   3          0.3  0.50       0.95       0.75       0.6               150    
    ##   3          0.3  0.50       0.95       0.75       0.8                50    
    ##   3          0.3  0.50       0.95       0.75       0.8               100    
    ##   3          0.3  0.50       0.95       0.75       0.8               150    
    ##   3          0.3  0.50       0.95       1.00       0.6                50    
    ##   3          0.3  0.50       0.95       1.00       0.6               100    
    ##   3          0.3  0.50       0.95       1.00       0.6               150    
    ##   3          0.3  0.50       0.95       1.00       0.8                50    
    ##   3          0.3  0.50       0.95       1.00       0.8               100    
    ##   3          0.3  0.50       0.95       1.00       0.8               150    
    ##   3          0.4  0.01       0.05       0.50       0.6                50    
    ##   3          0.4  0.01       0.05       0.50       0.6               100    
    ##   3          0.4  0.01       0.05       0.50       0.6               150    
    ##   3          0.4  0.01       0.05       0.50       0.8                50    
    ##   3          0.4  0.01       0.05       0.50       0.8               100    
    ##   3          0.4  0.01       0.05       0.50       0.8               150    
    ##   3          0.4  0.01       0.05       0.75       0.6                50    
    ##   3          0.4  0.01       0.05       0.75       0.6               100    
    ##   3          0.4  0.01       0.05       0.75       0.6               150    
    ##   3          0.4  0.01       0.05       0.75       0.8                50    
    ##   3          0.4  0.01       0.05       0.75       0.8               100    
    ##   3          0.4  0.01       0.05       0.75       0.8               150    
    ##   3          0.4  0.01       0.05       1.00       0.6                50    
    ##   3          0.4  0.01       0.05       1.00       0.6               100    
    ##   3          0.4  0.01       0.05       1.00       0.6               150    
    ##   3          0.4  0.01       0.05       1.00       0.8                50    
    ##   3          0.4  0.01       0.05       1.00       0.8               100    
    ##   3          0.4  0.01       0.05       1.00       0.8               150    
    ##   3          0.4  0.01       0.95       0.50       0.6                50    
    ##   3          0.4  0.01       0.95       0.50       0.6               100    
    ##   3          0.4  0.01       0.95       0.50       0.6               150    
    ##   3          0.4  0.01       0.95       0.50       0.8                50    
    ##   3          0.4  0.01       0.95       0.50       0.8               100    
    ##   3          0.4  0.01       0.95       0.50       0.8               150    
    ##   3          0.4  0.01       0.95       0.75       0.6                50    
    ##   3          0.4  0.01       0.95       0.75       0.6               100    
    ##   3          0.4  0.01       0.95       0.75       0.6               150    
    ##   3          0.4  0.01       0.95       0.75       0.8                50    
    ##   3          0.4  0.01       0.95       0.75       0.8               100    
    ##   3          0.4  0.01       0.95       0.75       0.8               150    
    ##   3          0.4  0.01       0.95       1.00       0.6                50    
    ##   3          0.4  0.01       0.95       1.00       0.6               100    
    ##   3          0.4  0.01       0.95       1.00       0.6               150    
    ##   3          0.4  0.01       0.95       1.00       0.8                50    
    ##   3          0.4  0.01       0.95       1.00       0.8               100    
    ##   3          0.4  0.01       0.95       1.00       0.8               150    
    ##   3          0.4  0.50       0.05       0.50       0.6                50    
    ##   3          0.4  0.50       0.05       0.50       0.6               100    
    ##   3          0.4  0.50       0.05       0.50       0.6               150    
    ##   3          0.4  0.50       0.05       0.50       0.8                50    
    ##   3          0.4  0.50       0.05       0.50       0.8               100    
    ##   3          0.4  0.50       0.05       0.50       0.8               150    
    ##   3          0.4  0.50       0.05       0.75       0.6                50    
    ##   3          0.4  0.50       0.05       0.75       0.6               100    
    ##   3          0.4  0.50       0.05       0.75       0.6               150    
    ##   3          0.4  0.50       0.05       0.75       0.8                50    
    ##   3          0.4  0.50       0.05       0.75       0.8               100    
    ##   3          0.4  0.50       0.05       0.75       0.8               150    
    ##   3          0.4  0.50       0.05       1.00       0.6                50    
    ##   3          0.4  0.50       0.05       1.00       0.6               100    
    ##   3          0.4  0.50       0.05       1.00       0.6               150    
    ##   3          0.4  0.50       0.05       1.00       0.8                50    
    ##   3          0.4  0.50       0.05       1.00       0.8               100    
    ##   3          0.4  0.50       0.05       1.00       0.8               150    
    ##   3          0.4  0.50       0.95       0.50       0.6                50    
    ##   3          0.4  0.50       0.95       0.50       0.6               100    
    ##   3          0.4  0.50       0.95       0.50       0.6               150    
    ##   3          0.4  0.50       0.95       0.50       0.8                50    
    ##   3          0.4  0.50       0.95       0.50       0.8               100    
    ##   3          0.4  0.50       0.95       0.50       0.8               150    
    ##   3          0.4  0.50       0.95       0.75       0.6                50    
    ##   3          0.4  0.50       0.95       0.75       0.6               100    
    ##   3          0.4  0.50       0.95       0.75       0.6               150    
    ##   3          0.4  0.50       0.95       0.75       0.8                50    
    ##   3          0.4  0.50       0.95       0.75       0.8               100    
    ##   3          0.4  0.50       0.95       0.75       0.8               150    
    ##   3          0.4  0.50       0.95       1.00       0.6                50    
    ##   3          0.4  0.50       0.95       1.00       0.6               100    
    ##   3          0.4  0.50       0.95       1.00       0.6               150    
    ##   3          0.4  0.50       0.95       1.00       0.8                50    
    ##   3          0.4  0.50       0.95       1.00       0.8               100    
    ##   3          0.4  0.50       0.95       1.00       0.8               150    
    ##   RMSE      Rsquared   MAE      
    ##   3.055967  0.9000135  2.7648573
    ##   1.820264  0.9037289  1.4858884
    ##   1.252259  0.9021310  0.9581254
    ##   2.757668  0.8403595  2.3176718
    ##   2.079071  0.8498509  1.6513595
    ##   1.530498  0.8514515  1.1805515
    ##   2.950013  0.8790168  2.6514735
    ##   1.637737  0.8904010  1.3402383
    ##   1.348116  0.8943597  1.0285733
    ##   2.808110  0.8660776  2.4320261
    ##   1.723934  0.8750248  1.3974053
    ##   1.393275  0.8795202  1.0812995
    ##   3.413517  0.8827915  3.0842912
    ##   1.851453  0.8890886  1.4890236
    ##   1.335000  0.8914112  1.0539870
    ##   2.652472  0.8625677  2.2796378
    ##   1.849269  0.8673331  1.5214095
    ##   1.457794  0.8689167  1.1096396
    ##   1.303826  0.8873040  1.0533943
    ##   1.323103  0.8920538  1.0471689
    ##   1.296736  0.8939078  1.0196386
    ##   1.515031  0.8506728  1.1728326
    ##   1.459734  0.8568478  1.1278159
    ##   1.442302  0.8602665  1.1049699
    ##   1.341146  0.8893922  1.0244965
    ##   1.326907  0.8945773  1.0131727
    ##   1.330173  0.8967410  0.9968523
    ##   1.542821  0.8377666  1.1269816
    ##   1.474987  0.8593149  1.0654983
    ##   1.446069  0.8657031  1.0183511
    ##   1.590548  0.8701865  1.2958101
    ##   1.584776  0.8723852  1.2923663
    ##   1.468000  0.8706325  1.1868871
    ##   1.454594  0.8686443  1.1546482
    ##   1.460694  0.8683987  1.1344179
    ##   1.482228  0.8669398  1.1331718
    ##   6.417527  0.7636579  5.8367097
    ##   3.382636  0.8210347  2.6881877
    ##   1.873248  0.8151722  1.4588901
    ##   6.271322  0.7234238  5.7196103
    ##   3.248990  0.7441303  2.5267901
    ##   2.019146  0.7448486  1.4876863
    ##   6.016787  0.7899848  5.5467605
    ##   3.068340  0.8202722  2.4693977
    ##   1.776907  0.8211547  1.3864169
    ##   6.306316  0.7607894  5.7876335
    ##   3.247142  0.8173100  2.6158195
    ##   1.883154  0.8155206  1.4206526
    ##   5.978915  0.7674339  5.5103183
    ##   3.038212  0.7675579  2.3963723
    ##   1.856742  0.7789365  1.3763448
    ##   5.673535  0.7623888  5.2292415
    ##   2.900968  0.7618137  2.3126366
    ##   1.820724  0.7820835  1.3706204
    ##   1.509546  0.8537131  1.1336906
    ##   1.356838  0.8748174  0.9769748
    ##   1.381219  0.8762209  0.9509667
    ##   1.370987  0.8888246  1.0755456
    ##   1.350188  0.8985260  1.0790614
    ##   1.298793  0.9079752  1.0137359
    ##   1.380560  0.8831784  1.0820397
    ##   1.321462  0.8904776  1.0240180
    ##   1.386789  0.8900651  1.0800140
    ##   1.423736  0.8690063  1.1081065
    ##   1.388504  0.8749247  1.0943955
    ##   1.344854  0.8795908  1.0335846
    ##   1.435836  0.8744963  1.1043606
    ##   1.409872  0.8795269  1.0804692
    ##   1.418127  0.8801789  1.0576365
    ##   1.493679  0.8612408  1.1389278
    ##   1.502500  0.8679502  1.1762129
    ##   1.558951  0.8646827  1.2028446
    ##   4.313119  0.8292984  3.9676119
    ##   2.943035  0.8409163  2.5451695
    ##   1.490016  0.8556810  1.1359279
    ##   3.783641  0.8869508  3.4483997
    ##   1.604444  0.8867943  1.2370528
    ##   1.210995  0.8981220  0.8787401
    ##   3.660738  0.8853816  3.3027759
    ##   2.339739  0.8994533  2.0317675
    ##   1.246345  0.9026318  0.9173140
    ##   3.618949  0.9005208  3.3240581
    ##   2.771591  0.9067390  2.4460030
    ##   1.281739  0.9073547  1.0450364
    ##   4.198290  0.8655508  3.8927375
    ##   2.321965  0.8660702  1.9328807
    ##   1.549721  0.8655843  1.1933595
    ##   3.433763  0.8811360  3.1573161
    ##   2.106249  0.8722608  1.7357143
    ##   1.483361  0.8673934  1.1534573
    ##   1.598774  0.8693807  1.2448773
    ##   1.516141  0.8808723  1.1618295
    ##   1.419395  0.8795236  1.0134527
    ##   1.510656  0.8484124  1.2120048
    ##   1.483963  0.8554161  1.1518438
    ##   1.449426  0.8583375  1.0990792
    ##   1.286838  0.8997850  1.0880885
    ##   1.306001  0.9030542  1.0605884
    ##   1.302590  0.9030100  1.0338814
    ##   1.251788  0.9013813  1.0354067
    ##   1.238582  0.9073193  1.0207595
    ##   1.246353  0.9066826  0.9986772
    ##   1.251660  0.9056111  1.0063963
    ##   1.311213  0.8993378  1.0239378
    ##   1.328000  0.8973048  1.0203319
    ##   1.483439  0.8595847  1.1273634
    ##   1.523824  0.8557680  1.1647268
    ##   1.551742  0.8534111  1.1807706
    ##   7.196928  0.8092031  6.7072676
    ##   3.609650  0.8054643  2.9921757
    ##   1.841995  0.7944489  1.4219860
    ##   7.488914  0.7953585  6.9587813
    ##   3.783224  0.7957957  3.0823679
    ##   1.768049  0.8354950  1.3218730
    ##   7.035539  0.7686390  6.5727394
    ##   3.448538  0.7911721  2.8348071
    ##   1.729616  0.7998248  1.3049613
    ##   7.189230  0.7780612  6.7128286
    ##   3.674622  0.7687460  3.0047366
    ##   1.679197  0.8148013  1.2231098
    ##   7.029865  0.7686215  6.6110117
    ##   3.387256  0.7725919  2.8152456
    ##   1.803865  0.7950407  1.3052074
    ##   7.200241  0.7522799  6.7418597
    ##   3.495275  0.7553957  2.8758349
    ##   1.993276  0.7502003  1.4998994
    ##   1.436702  0.8903511  1.1016128
    ##   1.358598  0.8982225  1.0377007
    ##   1.345567  0.8999035  1.0179123
    ##   1.344927  0.8847463  1.0326685
    ##   1.352054  0.8915282  1.0186378
    ##   1.355463  0.8902877  1.0218799
    ##   1.434264  0.8687686  1.1142834
    ##   1.380243  0.8730638  1.0905171
    ##   1.371153  0.8732429  1.0898799
    ##   1.497420  0.8683062  1.1778546
    ##   1.442755  0.8724471  1.1071987
    ##   1.429371  0.8709927  1.0908033
    ##   1.497073  0.8624074  1.1721831
    ##   1.550183  0.8648646  1.2269511
    ##   1.589180  0.8635064  1.2440141
    ##   1.467534  0.8670242  1.1281204
    ##   1.524790  0.8615844  1.1472763
    ##   1.551106  0.8581792  1.1602661
    ##   3.966910  0.8929673  3.6744974
    ##   2.371895  0.8998722  2.0401818
    ##   1.330267  0.8963680  1.0242422
    ##   3.132675  0.8795347  2.7890504
    ##   1.727426  0.8920044  1.3596295
    ##   1.306210  0.8904724  1.0098190
    ##   3.192430  0.8643772  2.8525861
    ##   1.976870  0.8684231  1.6913270
    ##   1.448921  0.8724731  1.0542331
    ##   3.459133  0.8959341  3.1831077
    ##   2.141870  0.8935634  1.7889318
    ##   1.324758  0.8933303  1.0128217
    ##   2.698627  0.8697257  2.3478730
    ##   1.579334  0.8726756  1.2416939
    ##   1.423722  0.8771376  1.0738573
    ##   3.114739  0.8336937  2.7555305
    ##   2.060377  0.8425157  1.6780918
    ##   1.517059  0.8533505  1.1511499
    ##   1.380318  0.9077974  1.0858257
    ##   1.181025  0.9074393  0.8739042
    ##   1.181217  0.9085538  0.8653038
    ##   1.514988  0.8438503  1.1910283
    ##   1.447928  0.8506211  1.1146387
    ##   1.457807  0.8495358  1.1233315
    ##   1.179664  0.9137191  0.9586270
    ##   1.157448  0.9179116  0.9050573
    ##   1.181557  0.9154390  0.9397372
    ##   1.571438  0.8504648  1.1199833
    ##   1.585176  0.8498170  1.1140804
    ##   1.548491  0.8497083  1.0736816
    ##   1.492034  0.8819682  1.1783949
    ##   1.433428  0.8845301  1.1022400
    ##   1.418501  0.8836612  1.0905716
    ##   1.572627  0.8521909  1.2299411
    ##   1.552093  0.8525761  1.1623051
    ##   1.552199  0.8525420  1.1448888
    ##   6.700305  0.8159375  6.0839580
    ##   3.661011  0.8077478  2.7965454
    ##   2.005931  0.8132586  1.5604685
    ##   6.556447  0.7487532  5.9344758
    ##   3.354606  0.7626562  2.6611875
    ##   2.018640  0.7731850  1.4411512
    ##   6.029842  0.7300569  5.5121142
    ##   3.089267  0.7728541  2.3843383
    ##   1.868203  0.7873484  1.4230598
    ##   6.366814  0.7926978  5.8956435
    ##   3.177609  0.7968380  2.5986892
    ##   1.781194  0.8162289  1.2910081
    ##   6.067229  0.8047185  5.5959100
    ##   2.978001  0.8347142  2.4424756
    ##   1.720107  0.8389813  1.2281539
    ##   5.969475  0.7907198  5.5185366
    ##   2.996898  0.7932924  2.4148728
    ##   1.783244  0.8051511  1.2899424
    ##   1.286440  0.8979276  1.0495543
    ##   1.239225  0.9058869  0.9682915
    ##   1.220892  0.9070822  0.9383113
    ##   1.677045  0.8269380  1.2896095
    ##   1.619553  0.8364095  1.2565813
    ##   1.627186  0.8391667  1.2610946
    ##   1.476234  0.8810295  1.0941158
    ##   1.469470  0.8787719  1.0910864
    ##   1.456025  0.8800211  1.0746158
    ##   1.537428  0.8508245  1.1665875
    ##   1.517177  0.8533529  1.1238735
    ##   1.517008  0.8530467  1.1280251
    ##   1.461449  0.8799787  1.1261836
    ##   1.448326  0.8791982  1.0901417
    ##   1.441328  0.8785290  1.0739431
    ##   1.505021  0.8551065  1.1414257
    ##   1.476242  0.8556940  1.0825584
    ##   1.480043  0.8558666  1.0609895
    ##   4.093937  0.8441414  3.7188374
    ##   2.847624  0.8467801  2.4545101
    ##   1.439225  0.8549383  1.0959854
    ##   3.730734  0.8496026  3.3772132
    ##   2.248814  0.8526196  1.8161666
    ##   1.508672  0.8549624  1.1079609
    ##   4.050164  0.9006752  3.7814091
    ##   2.810956  0.8888309  2.5029396
    ##   1.436792  0.8879596  1.1226729
    ##   4.008059  0.8325163  3.5315452
    ##   2.309795  0.8440014  1.8869511
    ##   1.567167  0.8478254  1.1421647
    ##   3.595138  0.8882247  3.2537345
    ##   2.152283  0.8826090  1.7332666
    ##   1.372522  0.8846442  1.0218351
    ##   3.771594  0.8682107  3.4087601
    ##   2.842808  0.8684932  2.4157926
    ##   1.440516  0.8680678  1.0812781
    ##   1.995617  0.8125111  1.6349623
    ##   1.939622  0.8110224  1.5704018
    ##   1.686874  0.8108746  1.3102865
    ##   1.496834  0.8584814  1.1255521
    ##   1.475777  0.8655528  1.1082279
    ##   1.484053  0.8647116  1.1143621
    ##   1.358944  0.9009728  1.0816631
    ##   1.362804  0.8997058  1.0506902
    ##   1.360848  0.8996355  1.0434669
    ##   1.537169  0.8628447  1.2397621
    ##   1.430386  0.8599000  1.1256296
    ##   1.439610  0.8590607  1.1327318
    ##   1.484976  0.8723936  1.1427182
    ##   1.495571  0.8720288  1.1421912
    ##   1.477107  0.8740032  1.1248561
    ##   1.488525  0.8641901  1.0960813
    ##   1.492537  0.8647695  1.0869621
    ##   1.497302  0.8660097  1.1105760
    ##   7.357298  0.8230319  6.8801688
    ##   3.739786  0.7981943  3.1447967
    ##   1.924139  0.8071223  1.4630972
    ##   7.822165  0.7942663  7.2594442
    ##   4.142036  0.7780487  3.4146835
    ##   2.080546  0.7675455  1.6207217
    ##   6.995316  0.8132853  6.5361133
    ##   3.341629  0.8235121  2.7738059
    ##   1.661294  0.8253470  1.2747946
    ##   7.012535  0.8218197  6.6101240
    ##   3.337245  0.8323047  2.8357435
    ##   1.534282  0.8432092  1.1394858
    ##   6.981673  0.7802067  6.5567496
    ##   3.419677  0.8110403  2.8582758
    ##   1.813074  0.8066140  1.3608536
    ##   7.028438  0.7722975  6.6231696
    ##   3.349058  0.7909744  2.7696484
    ##   1.784956  0.7985256  1.2691780
    ##   1.525496  0.8694143  1.2132089
    ##   1.418169  0.8713441  1.1112138
    ##   1.386221  0.8700874  1.0757693
    ##   1.511508  0.8666258  1.1399103
    ##   1.452666  0.8665901  1.0406264
    ##   1.438235  0.8666864  1.0140074
    ##   1.669324  0.8322947  1.2138931
    ##   1.640239  0.8353984  1.1960247
    ##   1.639611  0.8361735  1.1734104
    ##   1.526978  0.8724843  1.1981307
    ##   1.493150  0.8718300  1.1643748
    ##   1.459293  0.8716918  1.1131694
    ##   1.498271  0.8742863  1.1257462
    ##   1.496271  0.8721328  1.1503852
    ##   1.504062  0.8721887  1.1461183
    ##   1.495723  0.8678719  1.1244254
    ##   1.416735  0.8670354  1.0230225
    ##   1.438241  0.8672168  1.0549184
    ##   3.253206  0.8832947  2.8594187
    ##   1.929550  0.8880916  1.5335199
    ##   1.347383  0.8917539  1.0489223
    ##   2.738524  0.8320888  2.2248221
    ##   2.115746  0.8316091  1.6485243
    ##   1.595270  0.8320950  1.2636652
    ##   2.963346  0.8732535  2.6580113
    ##   1.945376  0.8765726  1.6010908
    ##   1.442752  0.8769325  1.1640478
    ##   3.459074  0.9072588  3.1448568
    ##   1.937282  0.9073758  1.5580191
    ##   1.245218  0.9048652  0.9218475
    ##   3.074816  0.8802136  2.6474060
    ##   1.751400  0.8850032  1.3882707
    ##   1.332071  0.8815654  1.0070228
    ##   2.959116  0.8828179  2.6215942
    ##   1.862543  0.8815637  1.5368409
    ##   1.410914  0.8787470  1.0370301
    ##   1.637639  0.8228355  1.2383385
    ##   1.613859  0.8235389  1.2168866
    ##   1.590840  0.8260432  1.2075634
    ##   1.507151  0.8594343  1.2175004
    ##   1.477477  0.8678931  1.1806198
    ##   1.476379  0.8679509  1.1766343
    ##   1.493928  0.8646485  1.1643090
    ##   1.485303  0.8654072  1.1549923
    ##   1.477965  0.8662906  1.1269200
    ##   1.463499  0.8611273  1.1008780
    ##   1.463082  0.8631984  1.0999531
    ##   1.452815  0.8650578  1.0689258
    ##   1.362914  0.8815453  1.0054782
    ##   1.342418  0.8816116  1.0308093
    ##   1.369276  0.8824594  1.0397175
    ##   1.375671  0.8868624  1.0088579
    ##   1.370229  0.8874488  1.0246732
    ##   1.371429  0.8876159  1.0275303
    ##   7.051795  0.7868001  6.4047335
    ##   3.840160  0.8197727  3.0231375
    ##   2.014447  0.8543709  1.5058917
    ##   6.491201  0.7534709  5.9029531
    ##   3.374804  0.7843102  2.6761779
    ##   1.976962  0.8097082  1.5036455
    ##   6.315117  0.7811008  5.7973140
    ##   3.135374  0.8199133  2.5730735
    ##   1.742247  0.8272538  1.2829419
    ##   6.202388  0.8095338  5.7567059
    ##   3.299595  0.7999594  2.6399449
    ##   1.762222  0.8099167  1.3485252
    ##   6.267856  0.7732573  5.7851493
    ##   3.136755  0.7946838  2.5008071
    ##   1.862671  0.7853713  1.3879356
    ##   6.141313  0.7619107  5.6781532
    ##   3.159727  0.7675385  2.5844097
    ##   1.809736  0.7998000  1.3048965
    ##   1.557701  0.8554660  1.1620955
    ##   1.508286  0.8588624  1.0831488
    ##   1.486274  0.8610934  1.0747780
    ##   1.369816  0.8821959  1.0632989
    ##   1.312489  0.8870656  1.0224442
    ##   1.324325  0.8862308  1.0207503
    ##   1.412968  0.8642647  1.0301942
    ##   1.386825  0.8643410  1.0055099
    ##   1.405759  0.8639678  1.0432582
    ##   1.367816  0.8844055  1.0229796
    ##   1.345999  0.8860182  0.9871598
    ##   1.340910  0.8860589  0.9737825
    ##   1.416976  0.8791745  1.0534703
    ##   1.413906  0.8787612  1.0449943
    ##   1.422504  0.8788872  1.0615452
    ##   1.445143  0.8673723  1.1137369
    ##   1.447776  0.8684457  1.1154573
    ##   1.445630  0.8684066  1.1224965
    ##   3.310027  0.8165719  2.8638840
    ##   1.919510  0.8310880  1.4574207
    ##   1.638821  0.8349176  1.2205172
    ##   3.600081  0.8902573  3.3438085
    ##   2.082947  0.9006932  1.8016364
    ##   1.289664  0.9070926  0.9630817
    ##   4.072784  0.8473434  3.7334275
    ##   2.518205  0.8494267  2.1088833
    ##   1.538757  0.8468881  1.2124590
    ##   3.553523  0.8730397  3.2785777
    ##   1.889626  0.8834644  1.5992802
    ##   1.378319  0.8854588  1.0126956
    ##   2.542486  0.8519382  2.2349148
    ##   1.759197  0.8523096  1.3900811
    ##   1.561309  0.8565549  1.1541424
    ##   3.229822  0.8686721  2.9642105
    ##   1.799352  0.8642927  1.4733110
    ##   1.500725  0.8656590  1.0811041
    ##   1.498647  0.8742544  1.1320174
    ##   1.517512  0.8724495  1.1436227
    ##   1.501157  0.8713467  1.1403728
    ##   1.462624  0.8696248  1.0557631
    ##   1.463482  0.8716220  1.0500751
    ##   1.459273  0.8723670  1.0475281
    ##   1.674953  0.8835315  1.3351638
    ##   1.412305  0.8856459  1.1029895
    ##   1.413421  0.8856967  1.1041681
    ##   1.522823  0.8616820  1.1290876
    ##   1.507628  0.8592507  1.1033176
    ##   1.516363  0.8613600  1.1377764
    ##   1.675901  0.8720553  1.2719073
    ##   1.676597  0.8718193  1.2719974
    ##   1.417447  0.8720546  1.0172883
    ##   1.653480  0.8439938  1.2590931
    ##   1.613684  0.8470007  1.2260229
    ##   1.546835  0.8473088  1.1774681
    ##   7.380660  0.8606398  6.9059713
    ##   3.601662  0.8462289  3.0127896
    ##   1.664784  0.8120023  1.2510289
    ##   7.256427  0.7742039  6.7199961
    ##   3.532656  0.8131085  2.9256477
    ##   1.843613  0.8097836  1.3716297
    ##   7.098095  0.8694349  6.6948668
    ##   3.407469  0.8493884  2.9159711
    ##   1.667867  0.8445193  1.2196434
    ##   7.054540  0.7592382  6.5815887
    ##   3.220477  0.7918038  2.6596774
    ##   1.734500  0.8155387  1.3082802
    ##   6.981369  0.8152075  6.6001651
    ##   3.306748  0.8460286  2.8171758
    ##   1.658107  0.8379375  1.2636435
    ##   6.931800  0.7718776  6.4826024
    ##   3.199404  0.7872988  2.6503002
    ##   1.698703  0.8127111  1.3320731
    ##   1.698285  0.8715767  1.2509969
    ##   1.599028  0.8730942  1.1946769
    ##   1.471998  0.8729989  1.1183580
    ##   1.629724  0.8358482  1.2303675
    ##   1.513073  0.8465286  1.1198333
    ##   1.502352  0.8464054  1.1065701
    ##   1.510398  0.8832938  1.0667968
    ##   1.440653  0.8836271  1.0142735
    ##   1.409040  0.8831756  0.9803663
    ##   1.322619  0.8867079  1.0701128
    ##   1.308917  0.8863829  1.0255567
    ##   1.338712  0.8860908  1.0488184
    ##   1.532854  0.8747257  1.1511587
    ##   1.500857  0.8746428  1.1436008
    ##   1.504016  0.8746110  1.1462268
    ##   1.510277  0.8575549  1.1254893
    ##   1.514450  0.8575546  1.1359866
    ##   1.528563  0.8574666  1.1454024
    ## 
    ## Tuning parameter 'gamma' was held constant at a value of 0
    ## Tuning
    ##  parameter 'min_child_weight' was held constant at a value of 1
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final values used for the model were nrounds = 100, max_depth = 2, eta
    ##  = 0.3, gamma = 0, subsample = 0.75, colsample_bytree = 0.6, rate_drop =
    ##  0.01, skip_drop = 0.95 and min_child_weight = 1.

## 7.7 Evaluating models

Our main objective for this project is to deliver a model that will be
used to predict the Energy Consumption of electrical cars.

In this case, we are concerned in reducing the error of our model. Three
metrics will be used to evaluate the result: **R²** or coefficient of
determination, which is the proportion of the variance for a dependent
variable that is explained by independend variables; **MAE** or mean
absolute error, which is the average absolute error between actual and
predicted values; **RMSE** or root mean square error, which is the
starndard deviation of the residuals (prediction errors).

``` r
predict_models_1 <- predict(model_trained_1, newdata = test)

for (i in 1:length(predict_models_1)){
  
  print(names(predict_models_1[i]))
  
  result <- postResample(pred = predict_models_1[[i]], obs = test$Consumption)
  
  print(result)
  
  print("------------------------------------------------------------------------------")
}
```

    ## [1] "lm"
    ##      RMSE  Rsquared       MAE 
    ## 1.5432117 0.9353417 1.1226395 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "ridge"
    ##      RMSE  Rsquared       MAE 
    ## 1.5432117 0.9353417 1.1226395 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "rf"
    ##      RMSE  Rsquared       MAE 
    ## 1.4281237 0.9267261 1.0843771 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "xgbDART"
    ##      RMSE  Rsquared       MAE 
    ## 1.2775103 0.9345743 1.0041359 
    ## [1] "------------------------------------------------------------------------------"

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

## 7.8 Ranking features by importance

It is interesting now to evaluate wich features affect most our
prediction model, and eventually discard some.

``` r
chosen_model_1 <- train(`Consumption` ~ ., data = train_validation, 
                        method = "lm",
                        trControl = train.control,
                        metric = 'Rsquared',
                        verbosity = 0)

importance <- varImp(chosen_model_1, scale=TRUE)

plot(importance)
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

``` r
summary(chosen_model_1)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat, verbosity = 0)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -2.3834 -0.6295  0.1066  0.4983  1.8515 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)      18.94545    0.20446  92.659  < 2e-16 ***
    ## Power             3.11247    1.28634   2.420  0.02425 *  
    ## Torque            0.91439    0.90110   1.015  0.32125    
    ## Battery_Capacity -0.70760    0.65356  -1.083  0.29068    
    ## Gross_Weight      1.53453    0.86037   1.784  0.08830 .  
    ## Load_Capacity     1.31964    0.40540   3.255  0.00363 ** 
    ## Tire_Size        -1.62657    0.55375  -2.937  0.00762 ** 
    ## Max_Speed        -1.59358    0.79905  -1.994  0.05865 .  
    ## Boot_Capacity     0.09905    0.67434   0.147  0.88457    
    ## Drive_Type_2WD   -0.83826    0.43316  -1.935  0.06592 .  
    ## Drive_Type_4WD    0.46029    0.69283   0.664  0.51336    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.175 on 22 degrees of freedom
    ## Multiple R-squared:  0.9432, Adjusted R-squared:  0.9174 
    ## F-statistic: 36.52 on 10 and 22 DF,  p-value: 2.194e-11

As we can see, the feature `Boot_Capacity` seems to have a smaller
effect on the prediction model. For this reason, we will not consider it
in our final model.

## 7.9 Optmizing model

``` r
opt_model_1 <- train(`Consumption` ~ ., data = train_validation[, -8], 
                     method = "lm",
                     trControl = train.control,
                     metric = 'Rsquared')

predict_opt_1 <- predict(opt_model_1, newdata = test)

result_df_1 <- postResample(pred = predict_opt_1, obs = test$Consumption)

result_df_1
```

    ##      RMSE  Rsquared       MAE 
    ## 1.5457831 0.9375769 1.1278111

We were able to keep up our performance while we reduced dimensionality
of the final model.

# 8. Second Analysis: Inputting missing values

``` r
df_2 <- copy(dataset)
```

## 8.1 Treating `Gross_Weight`

In our previous exploratory analysis we could observe a high positive
correlation between `Gross_Weight` and `Minimal_Weight`. In fact, the
permissable gross weight is probably derived from the minimal empty
weight. Let’s start by calculating the ratio Gross_Weight /
Minimal_Weight:

``` r
relation_weight <- df_2[, "Gross_Weight"] / df_2[, "Minimal_Weight"]

relation_weight <- na.omit(relation_weight)

summary(relation_weight)
```

    ##   Gross_Weight  
    ##  Min.   :1.096  
    ##  1st Qu.:1.210  
    ##  Median :1.255  
    ##  Mean   :1.257  
    ##  3rd Qu.:1.298  
    ##  Max.   :1.427

``` r
relation_weight %>% apply(2, sd)
```

    ## Gross_Weight 
    ##    0.0704115

In fact, the mentioned ratio seems to be a constant value over the
observations: both mean and median are similar, while the standard
deviation is not relatively high. **We’ll use the mean of that ratio to
replace missing values in our `Gross_Weight` feature**.

``` r
na_values <- is.na(df_2[, "Gross_Weight"])


for (i in which(na_values, arr.ind = F)){
  
  df_2[i, "Gross_Weight"] <- df_2[i, "Minimal_Weight"] * mean(relation_weight$Gross_Weight)
}
```

## 8.2 Treating `Load_Capacity`

As for `Load_Capacity`, we can run a similar test:

``` r
relation_capacity <- df_2[, "Gross_Weight"] / df_2[, "Load_Capacity"]

relation_capacity <- na.omit(relation_capacity)

summary(relation_capacity)
```

    ##   Gross_Weight  
    ##  Min.   :2.661  
    ##  1st Qu.:4.046  
    ##  Median :4.433  
    ##  Mean   :4.465  
    ##  3rd Qu.:4.920  
    ##  Max.   :6.607

``` r
relation_capacity %>% apply(2, sd)
```

    ## Gross_Weight 
    ##    0.7170691

Although the standard deviation looks higher than the previous
situation, we decided to follow the same strategy, that is, **We’ll use
the mean of the ratio to replace missing values in the `Load_Capacity`
feature**

``` r
na_values <- is.na(df_2[, "Load_Capacity"])


for (i in which(na_values, arr.ind = F)){
  
  df_2[i, "Load_Capacity"] <- df_2[i, "Gross_Weight"] / mean(relation_capacity$Gross_Weight)
}
```

## 8.3 Treating `Consumption`

In order to replace missing values for the `Consumption` feature, we’ll
**make use of our prediction model trained in the first analysis**. We
need to remind to use the same predictor features and standardize the
data with the same values used before.

``` r
na_values <- is.na(df_2[, "Consumption"])

miss_values <- df_2[na_values, c("Power", "Torque", "Drive_Type", "Battery_Capacity", "Gross_Weight", "Load_Capacity", "Tire_Size", "Max_Speed")]

miss_values <- dummy_cols(miss_values, select_columns = 'Drive_Type')

miss_values <- miss_values[, -c(3, 10)]

colnames(miss_values)[8] <- "Drive_Type_2WD"

for (i in colnames(miss_values)){
  miss_values[, i] <-  (miss_values[, i] - train_means[, i])/train_std[, i]
}

predict_missing <- data.frame(predict(opt_model_1, newdata = miss_values))

rownames(predict_missing) <- c(10, 30, 40, 41, 42, 43, 44, 45, 46)

for (row in rownames(predict_missing)){
  
  df_2[row, "Consumption"] <- predict_missing[row, ]
}
```

## 8.4 Treating other NA values

We decided to **simply omit the remaining missing values**, as they
don’t represent a great part of our dataset anymore.

``` r
df_2 <- na.omit(df_2)

dim(df_2)
```

    ## [1] 50 25

## 8.5 Feature correlation

``` r
numerical_columns_2 <- c(4:6, 9:25)

df_numeric_2 <- df_2[, numerical_columns_2]

matrix_correlation <- cor(df_numeric_2)

cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

p.mat <- cor.mtest(df_numeric_2)

col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))

corrplot(matrix_correlation, method="color", col=col(200),  
         type="upper", order="hclust", 
         addCoef.col = "black",
         tl.col="black", tl.srt=90, 
         p.mat = p.mat, sig.level = 0.05, insig = "blank", 
         diag=TRUE)
```

![](electric_cars_prediction_files/figure-gfm/fig2-1.png)<!-- -->

## 8.6 Statistical tests for categorical features

``` r
shapiro_test <- shapiro.test(df_2$Consumption)
shapiro_test
```

    ## 
    ##  Shapiro-Wilk normality test
    ## 
    ## data:  df_2$Consumption
    ## W = 0.88628, p-value = 0.0001746

``` r
kruskal_test_1 <- kruskal.test(Consumption ~ Type_Brakes, data = df_2)
kruskal_test_1
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Type_Brakes
    ## Kruskal-Wallis chi-squared = 3.9971, df = 1, p-value = 0.04558

Interesting to note that, at this time, **the feature `Type_Brakes`
appears to have statistical significance**.

``` r
kruskal_test_2 <- kruskal.test(Consumption ~ Drive_Type, data = df_2)
kruskal_test_2
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Drive_Type
    ## Kruskal-Wallis chi-squared = 30.796, df = 2, p-value = 2.054e-07

``` r
kruskal_test_3 <- kruskal.test(Consumption ~ Brand, data = df_2)
kruskal_test_3
```

    ## 
    ##  Kruskal-Wallis rank sum test
    ## 
    ## data:  Consumption by Brand
    ## Kruskal-Wallis chi-squared = 37.462, df = 19, p-value = 0.006943

``` r
wilcox_test_2 <- pairwise.wilcox.test(df_2$Consumption,
                                      df_2$Drive_Type,
                                      p.adjust.method="bonferroni", exact = FALSE)
wilcox_test_2
```

    ## 
    ##  Pairwise comparisons using Wilcoxon rank sum test with continuity correction 
    ## 
    ## data:  df_2$Consumption and df_2$Drive_Type 
    ## 
    ##            2WD (front) 2WD (rear)
    ## 2WD (rear) 1           -         
    ## 4WD        2.6e-06     2.9e-05   
    ## 
    ## P value adjustment method: bonferroni

``` r
wilcox_test_3 <- pairwise.wilcox.test(df_2$Consumption,
                                      df_2$Brand,
                                      p.adjust.method="bonferroni", exact = FALSE)
wilcox_test_3
```

    ## 
    ##  Pairwise comparisons using Wilcoxon rank sum test with continuity correction 
    ## 
    ## data:  df_2$Consumption and df_2$Brand 
    ## 
    ##               Audi BMW Citroën DS Honda Hyundai Jaguar Kia Mazda Mercedes-Benz
    ## BMW           1    -   -       -  -     -       -      -   -     -            
    ## Citroën       1    1   -       -  -     -       -      -   -     -            
    ## DS            1    1   1       -  -     -       -      -   -     -            
    ## Honda         1    1   1       1  -     -       -      -   -     -            
    ## Hyundai       1    1   1       1  1     -       -      -   -     -            
    ## Jaguar        1    1   1       1  1     1       -      -   -     -            
    ## Kia           1    1   1       1  1     1       1      -   -     -            
    ## Mazda         1    1   1       1  1     1       1      1   -     -            
    ## Mercedes-Benz 1    1   1       1  1     1       1      1   1     -            
    ## Mini          1    1   1       1  1     1       1      1   1     1            
    ## Nissan        1    1   1       1  1     1       1      1   1     1            
    ## Opel          1    1   1       1  1     1       1      1   1     1            
    ## Peugeot       1    1   1       1  1     1       1      1   1     1            
    ## Porsche       1    1   1       1  1     1       1      1   1     1            
    ## Renault       1    1   1       1  1     1       1      1   1     1            
    ## Skoda         1    1   1       1  1     1       1      1   1     1            
    ## Smart         1    1   1       1  1     1       1      1   1     1            
    ## Tesla         1    1   1       1  1     1       1      1   1     1            
    ## Volkswagen    1    1   1       1  1     1       1      1   1     1            
    ##               Mini Nissan Opel Peugeot Porsche Renault Skoda Smart Tesla
    ## BMW           -    -      -    -       -       -       -     -     -    
    ## Citroën       -    -      -    -       -       -       -     -     -    
    ## DS            -    -      -    -       -       -       -     -     -    
    ## Honda         -    -      -    -       -       -       -     -     -    
    ## Hyundai       -    -      -    -       -       -       -     -     -    
    ## Jaguar        -    -      -    -       -       -       -     -     -    
    ## Kia           -    -      -    -       -       -       -     -     -    
    ## Mazda         -    -      -    -       -       -       -     -     -    
    ## Mercedes-Benz -    -      -    -       -       -       -     -     -    
    ## Mini          -    -      -    -       -       -       -     -     -    
    ## Nissan        1    -      -    -       -       -       -     -     -    
    ## Opel          1    1      -    -       -       -       -     -     -    
    ## Peugeot       1    1      1    -       -       -       -     -     -    
    ## Porsche       1    1      1    1       -       -       -     -     -    
    ## Renault       1    1      1    1       1       -       -     -     -    
    ## Skoda         1    1      1    1       1       1       -     -     -    
    ## Smart         1    1      1    1       1       1       1     -     -    
    ## Tesla         1    1      1    1       1       1       1     1     -    
    ## Volkswagen    1    1      1    1       1       1       1     1     1    
    ## 
    ## P value adjustment method: bonferroni

## 8.7 Splitting data

Since our dataframe has now more observations, we can utilize a bigger
part of it to training and validation. We’ll consider the `Type_Brakes`
as a predictor feature as well.

``` r
model_2 <- df_2[, c(5, 6, 7, 8, 9, 16, 17, 20:22, 25)]

model_2 <- dummy_cols(model_2, select_columns = 'Drive_Type')

model_2 <- dummy_cols(model_2, select_columns = 'Type_Brakes')


model_2 <- model_2[, -c(3, 4, 13, 16)]

colnames(model_2)[10] <- "Drive_Type_2WD"

colnames(model_2)[12] <- "Type_Brakes_disc"

set.seed(67)

sample <- sample.split(model_2$Consumption, SplitRatio = 0.85)
train_validation <- subset(model_2, sample == TRUE)
test <- subset(model_2, sample == FALSE)
```

``` r
dim(model_2)
```

    ## [1] 50 12

``` r
dim(train_validation)
```

    ## [1] 42 12

``` r
dim(test)
```

    ## [1]  8 12

## 8.8 Standardizing data

``` r
train_means <- data.frame(as.list(train_validation %>% apply(2, mean)))

train_std <- data.frame(as.list(train_validation %>% apply(2, sd)))

col_names <- names(train_validation[, -9])

for (i in col_names){
  train_validation[, i] <- (train_validation[, i] - train_means[, i])/train_std[, i]
  test[, i] <-  (test[, i] - train_means[, i])/train_std[, i]
}
```

## 8.9 Choosing and training models

``` r
models = c("lm", "ridge", "rf", "xgbDART")

model_trained_2 <- c()

set.seed(67) 

train.control <- trainControl(method = "cv", number = 5)

for (model in models){
  model_trained_2[[model]] <- train(`Consumption` ~ ., data = train_validation, 
                                    method = model,
                                    trControl = train.control,
                                    metric = 'Rsquared',
                                    verbosity = 0)
}

print(model_trained_2)
```

    ## $lm
    ## Linear Regression 
    ## 
    ## 42 samples
    ## 11 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 34, 34, 33, 33, 34 
    ## Resampling results:
    ## 
    ##   RMSE      Rsquared   MAE      
    ##   1.162899  0.9311437  0.8718665
    ## 
    ## Tuning parameter 'intercept' was held constant at a value of TRUE
    ## 
    ## $ridge
    ## Ridge Regression 
    ## 
    ## 42 samples
    ## 11 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 34, 33, 33, 34, 34 
    ## Resampling results across tuning parameters:
    ## 
    ##   lambda  RMSE      Rsquared   MAE      
    ##   0e+00   1.152863  0.9392437  0.8880512
    ##   1e-04   1.152779  0.9391709  0.8882911
    ##   1e-01   1.514042  0.8955198  1.2564122
    ## 
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final value used for the model was lambda = 0.
    ## 
    ## $rf
    ## Random Forest 
    ## 
    ## 42 samples
    ## 11 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 34, 33, 33, 34, 34 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared   MAE     
    ##    2    1.724854  0.8875994  1.186166
    ##    6    1.664923  0.8997408  1.174068
    ##   11    1.622460  0.9032856  1.192089
    ## 
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final value used for the model was mtry = 11.
    ## 
    ## $xgbDART
    ## eXtreme Gradient Boosting 
    ## 
    ## 42 samples
    ## 11 predictors
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 34, 34, 34, 33, 33 
    ## Resampling results across tuning parameters:
    ## 
    ##   max_depth  eta  rate_drop  skip_drop  subsample  colsample_bytree  nrounds
    ##   1          0.3  0.01       0.05       0.50       0.6                50    
    ##   1          0.3  0.01       0.05       0.50       0.6               100    
    ##   1          0.3  0.01       0.05       0.50       0.6               150    
    ##   1          0.3  0.01       0.05       0.50       0.8                50    
    ##   1          0.3  0.01       0.05       0.50       0.8               100    
    ##   1          0.3  0.01       0.05       0.50       0.8               150    
    ##   1          0.3  0.01       0.05       0.75       0.6                50    
    ##   1          0.3  0.01       0.05       0.75       0.6               100    
    ##   1          0.3  0.01       0.05       0.75       0.6               150    
    ##   1          0.3  0.01       0.05       0.75       0.8                50    
    ##   1          0.3  0.01       0.05       0.75       0.8               100    
    ##   1          0.3  0.01       0.05       0.75       0.8               150    
    ##   1          0.3  0.01       0.05       1.00       0.6                50    
    ##   1          0.3  0.01       0.05       1.00       0.6               100    
    ##   1          0.3  0.01       0.05       1.00       0.6               150    
    ##   1          0.3  0.01       0.05       1.00       0.8                50    
    ##   1          0.3  0.01       0.05       1.00       0.8               100    
    ##   1          0.3  0.01       0.05       1.00       0.8               150    
    ##   1          0.3  0.01       0.95       0.50       0.6                50    
    ##   1          0.3  0.01       0.95       0.50       0.6               100    
    ##   1          0.3  0.01       0.95       0.50       0.6               150    
    ##   1          0.3  0.01       0.95       0.50       0.8                50    
    ##   1          0.3  0.01       0.95       0.50       0.8               100    
    ##   1          0.3  0.01       0.95       0.50       0.8               150    
    ##   1          0.3  0.01       0.95       0.75       0.6                50    
    ##   1          0.3  0.01       0.95       0.75       0.6               100    
    ##   1          0.3  0.01       0.95       0.75       0.6               150    
    ##   1          0.3  0.01       0.95       0.75       0.8                50    
    ##   1          0.3  0.01       0.95       0.75       0.8               100    
    ##   1          0.3  0.01       0.95       0.75       0.8               150    
    ##   1          0.3  0.01       0.95       1.00       0.6                50    
    ##   1          0.3  0.01       0.95       1.00       0.6               100    
    ##   1          0.3  0.01       0.95       1.00       0.6               150    
    ##   1          0.3  0.01       0.95       1.00       0.8                50    
    ##   1          0.3  0.01       0.95       1.00       0.8               100    
    ##   1          0.3  0.01       0.95       1.00       0.8               150    
    ##   1          0.3  0.50       0.05       0.50       0.6                50    
    ##   1          0.3  0.50       0.05       0.50       0.6               100    
    ##   1          0.3  0.50       0.05       0.50       0.6               150    
    ##   1          0.3  0.50       0.05       0.50       0.8                50    
    ##   1          0.3  0.50       0.05       0.50       0.8               100    
    ##   1          0.3  0.50       0.05       0.50       0.8               150    
    ##   1          0.3  0.50       0.05       0.75       0.6                50    
    ##   1          0.3  0.50       0.05       0.75       0.6               100    
    ##   1          0.3  0.50       0.05       0.75       0.6               150    
    ##   1          0.3  0.50       0.05       0.75       0.8                50    
    ##   1          0.3  0.50       0.05       0.75       0.8               100    
    ##   1          0.3  0.50       0.05       0.75       0.8               150    
    ##   1          0.3  0.50       0.05       1.00       0.6                50    
    ##   1          0.3  0.50       0.05       1.00       0.6               100    
    ##   1          0.3  0.50       0.05       1.00       0.6               150    
    ##   1          0.3  0.50       0.05       1.00       0.8                50    
    ##   1          0.3  0.50       0.05       1.00       0.8               100    
    ##   1          0.3  0.50       0.05       1.00       0.8               150    
    ##   1          0.3  0.50       0.95       0.50       0.6                50    
    ##   1          0.3  0.50       0.95       0.50       0.6               100    
    ##   1          0.3  0.50       0.95       0.50       0.6               150    
    ##   1          0.3  0.50       0.95       0.50       0.8                50    
    ##   1          0.3  0.50       0.95       0.50       0.8               100    
    ##   1          0.3  0.50       0.95       0.50       0.8               150    
    ##   1          0.3  0.50       0.95       0.75       0.6                50    
    ##   1          0.3  0.50       0.95       0.75       0.6               100    
    ##   1          0.3  0.50       0.95       0.75       0.6               150    
    ##   1          0.3  0.50       0.95       0.75       0.8                50    
    ##   1          0.3  0.50       0.95       0.75       0.8               100    
    ##   1          0.3  0.50       0.95       0.75       0.8               150    
    ##   1          0.3  0.50       0.95       1.00       0.6                50    
    ##   1          0.3  0.50       0.95       1.00       0.6               100    
    ##   1          0.3  0.50       0.95       1.00       0.6               150    
    ##   1          0.3  0.50       0.95       1.00       0.8                50    
    ##   1          0.3  0.50       0.95       1.00       0.8               100    
    ##   1          0.3  0.50       0.95       1.00       0.8               150    
    ##   1          0.4  0.01       0.05       0.50       0.6                50    
    ##   1          0.4  0.01       0.05       0.50       0.6               100    
    ##   1          0.4  0.01       0.05       0.50       0.6               150    
    ##   1          0.4  0.01       0.05       0.50       0.8                50    
    ##   1          0.4  0.01       0.05       0.50       0.8               100    
    ##   1          0.4  0.01       0.05       0.50       0.8               150    
    ##   1          0.4  0.01       0.05       0.75       0.6                50    
    ##   1          0.4  0.01       0.05       0.75       0.6               100    
    ##   1          0.4  0.01       0.05       0.75       0.6               150    
    ##   1          0.4  0.01       0.05       0.75       0.8                50    
    ##   1          0.4  0.01       0.05       0.75       0.8               100    
    ##   1          0.4  0.01       0.05       0.75       0.8               150    
    ##   1          0.4  0.01       0.05       1.00       0.6                50    
    ##   1          0.4  0.01       0.05       1.00       0.6               100    
    ##   1          0.4  0.01       0.05       1.00       0.6               150    
    ##   1          0.4  0.01       0.05       1.00       0.8                50    
    ##   1          0.4  0.01       0.05       1.00       0.8               100    
    ##   1          0.4  0.01       0.05       1.00       0.8               150    
    ##   1          0.4  0.01       0.95       0.50       0.6                50    
    ##   1          0.4  0.01       0.95       0.50       0.6               100    
    ##   1          0.4  0.01       0.95       0.50       0.6               150    
    ##   1          0.4  0.01       0.95       0.50       0.8                50    
    ##   1          0.4  0.01       0.95       0.50       0.8               100    
    ##   1          0.4  0.01       0.95       0.50       0.8               150    
    ##   1          0.4  0.01       0.95       0.75       0.6                50    
    ##   1          0.4  0.01       0.95       0.75       0.6               100    
    ##   1          0.4  0.01       0.95       0.75       0.6               150    
    ##   1          0.4  0.01       0.95       0.75       0.8                50    
    ##   1          0.4  0.01       0.95       0.75       0.8               100    
    ##   1          0.4  0.01       0.95       0.75       0.8               150    
    ##   1          0.4  0.01       0.95       1.00       0.6                50    
    ##   1          0.4  0.01       0.95       1.00       0.6               100    
    ##   1          0.4  0.01       0.95       1.00       0.6               150    
    ##   1          0.4  0.01       0.95       1.00       0.8                50    
    ##   1          0.4  0.01       0.95       1.00       0.8               100    
    ##   1          0.4  0.01       0.95       1.00       0.8               150    
    ##   1          0.4  0.50       0.05       0.50       0.6                50    
    ##   1          0.4  0.50       0.05       0.50       0.6               100    
    ##   1          0.4  0.50       0.05       0.50       0.6               150    
    ##   1          0.4  0.50       0.05       0.50       0.8                50    
    ##   1          0.4  0.50       0.05       0.50       0.8               100    
    ##   1          0.4  0.50       0.05       0.50       0.8               150    
    ##   1          0.4  0.50       0.05       0.75       0.6                50    
    ##   1          0.4  0.50       0.05       0.75       0.6               100    
    ##   1          0.4  0.50       0.05       0.75       0.6               150    
    ##   1          0.4  0.50       0.05       0.75       0.8                50    
    ##   1          0.4  0.50       0.05       0.75       0.8               100    
    ##   1          0.4  0.50       0.05       0.75       0.8               150    
    ##   1          0.4  0.50       0.05       1.00       0.6                50    
    ##   1          0.4  0.50       0.05       1.00       0.6               100    
    ##   1          0.4  0.50       0.05       1.00       0.6               150    
    ##   1          0.4  0.50       0.05       1.00       0.8                50    
    ##   1          0.4  0.50       0.05       1.00       0.8               100    
    ##   1          0.4  0.50       0.05       1.00       0.8               150    
    ##   1          0.4  0.50       0.95       0.50       0.6                50    
    ##   1          0.4  0.50       0.95       0.50       0.6               100    
    ##   1          0.4  0.50       0.95       0.50       0.6               150    
    ##   1          0.4  0.50       0.95       0.50       0.8                50    
    ##   1          0.4  0.50       0.95       0.50       0.8               100    
    ##   1          0.4  0.50       0.95       0.50       0.8               150    
    ##   1          0.4  0.50       0.95       0.75       0.6                50    
    ##   1          0.4  0.50       0.95       0.75       0.6               100    
    ##   1          0.4  0.50       0.95       0.75       0.6               150    
    ##   1          0.4  0.50       0.95       0.75       0.8                50    
    ##   1          0.4  0.50       0.95       0.75       0.8               100    
    ##   1          0.4  0.50       0.95       0.75       0.8               150    
    ##   1          0.4  0.50       0.95       1.00       0.6                50    
    ##   1          0.4  0.50       0.95       1.00       0.6               100    
    ##   1          0.4  0.50       0.95       1.00       0.6               150    
    ##   1          0.4  0.50       0.95       1.00       0.8                50    
    ##   1          0.4  0.50       0.95       1.00       0.8               100    
    ##   1          0.4  0.50       0.95       1.00       0.8               150    
    ##   2          0.3  0.01       0.05       0.50       0.6                50    
    ##   2          0.3  0.01       0.05       0.50       0.6               100    
    ##   2          0.3  0.01       0.05       0.50       0.6               150    
    ##   2          0.3  0.01       0.05       0.50       0.8                50    
    ##   2          0.3  0.01       0.05       0.50       0.8               100    
    ##   2          0.3  0.01       0.05       0.50       0.8               150    
    ##   2          0.3  0.01       0.05       0.75       0.6                50    
    ##   2          0.3  0.01       0.05       0.75       0.6               100    
    ##   2          0.3  0.01       0.05       0.75       0.6               150    
    ##   2          0.3  0.01       0.05       0.75       0.8                50    
    ##   2          0.3  0.01       0.05       0.75       0.8               100    
    ##   2          0.3  0.01       0.05       0.75       0.8               150    
    ##   2          0.3  0.01       0.05       1.00       0.6                50    
    ##   2          0.3  0.01       0.05       1.00       0.6               100    
    ##   2          0.3  0.01       0.05       1.00       0.6               150    
    ##   2          0.3  0.01       0.05       1.00       0.8                50    
    ##   2          0.3  0.01       0.05       1.00       0.8               100    
    ##   2          0.3  0.01       0.05       1.00       0.8               150    
    ##   2          0.3  0.01       0.95       0.50       0.6                50    
    ##   2          0.3  0.01       0.95       0.50       0.6               100    
    ##   2          0.3  0.01       0.95       0.50       0.6               150    
    ##   2          0.3  0.01       0.95       0.50       0.8                50    
    ##   2          0.3  0.01       0.95       0.50       0.8               100    
    ##   2          0.3  0.01       0.95       0.50       0.8               150    
    ##   2          0.3  0.01       0.95       0.75       0.6                50    
    ##   2          0.3  0.01       0.95       0.75       0.6               100    
    ##   2          0.3  0.01       0.95       0.75       0.6               150    
    ##   2          0.3  0.01       0.95       0.75       0.8                50    
    ##   2          0.3  0.01       0.95       0.75       0.8               100    
    ##   2          0.3  0.01       0.95       0.75       0.8               150    
    ##   2          0.3  0.01       0.95       1.00       0.6                50    
    ##   2          0.3  0.01       0.95       1.00       0.6               100    
    ##   2          0.3  0.01       0.95       1.00       0.6               150    
    ##   2          0.3  0.01       0.95       1.00       0.8                50    
    ##   2          0.3  0.01       0.95       1.00       0.8               100    
    ##   2          0.3  0.01       0.95       1.00       0.8               150    
    ##   2          0.3  0.50       0.05       0.50       0.6                50    
    ##   2          0.3  0.50       0.05       0.50       0.6               100    
    ##   2          0.3  0.50       0.05       0.50       0.6               150    
    ##   2          0.3  0.50       0.05       0.50       0.8                50    
    ##   2          0.3  0.50       0.05       0.50       0.8               100    
    ##   2          0.3  0.50       0.05       0.50       0.8               150    
    ##   2          0.3  0.50       0.05       0.75       0.6                50    
    ##   2          0.3  0.50       0.05       0.75       0.6               100    
    ##   2          0.3  0.50       0.05       0.75       0.6               150    
    ##   2          0.3  0.50       0.05       0.75       0.8                50    
    ##   2          0.3  0.50       0.05       0.75       0.8               100    
    ##   2          0.3  0.50       0.05       0.75       0.8               150    
    ##   2          0.3  0.50       0.05       1.00       0.6                50    
    ##   2          0.3  0.50       0.05       1.00       0.6               100    
    ##   2          0.3  0.50       0.05       1.00       0.6               150    
    ##   2          0.3  0.50       0.05       1.00       0.8                50    
    ##   2          0.3  0.50       0.05       1.00       0.8               100    
    ##   2          0.3  0.50       0.05       1.00       0.8               150    
    ##   2          0.3  0.50       0.95       0.50       0.6                50    
    ##   2          0.3  0.50       0.95       0.50       0.6               100    
    ##   2          0.3  0.50       0.95       0.50       0.6               150    
    ##   2          0.3  0.50       0.95       0.50       0.8                50    
    ##   2          0.3  0.50       0.95       0.50       0.8               100    
    ##   2          0.3  0.50       0.95       0.50       0.8               150    
    ##   2          0.3  0.50       0.95       0.75       0.6                50    
    ##   2          0.3  0.50       0.95       0.75       0.6               100    
    ##   2          0.3  0.50       0.95       0.75       0.6               150    
    ##   2          0.3  0.50       0.95       0.75       0.8                50    
    ##   2          0.3  0.50       0.95       0.75       0.8               100    
    ##   2          0.3  0.50       0.95       0.75       0.8               150    
    ##   2          0.3  0.50       0.95       1.00       0.6                50    
    ##   2          0.3  0.50       0.95       1.00       0.6               100    
    ##   2          0.3  0.50       0.95       1.00       0.6               150    
    ##   2          0.3  0.50       0.95       1.00       0.8                50    
    ##   2          0.3  0.50       0.95       1.00       0.8               100    
    ##   2          0.3  0.50       0.95       1.00       0.8               150    
    ##   2          0.4  0.01       0.05       0.50       0.6                50    
    ##   2          0.4  0.01       0.05       0.50       0.6               100    
    ##   2          0.4  0.01       0.05       0.50       0.6               150    
    ##   2          0.4  0.01       0.05       0.50       0.8                50    
    ##   2          0.4  0.01       0.05       0.50       0.8               100    
    ##   2          0.4  0.01       0.05       0.50       0.8               150    
    ##   2          0.4  0.01       0.05       0.75       0.6                50    
    ##   2          0.4  0.01       0.05       0.75       0.6               100    
    ##   2          0.4  0.01       0.05       0.75       0.6               150    
    ##   2          0.4  0.01       0.05       0.75       0.8                50    
    ##   2          0.4  0.01       0.05       0.75       0.8               100    
    ##   2          0.4  0.01       0.05       0.75       0.8               150    
    ##   2          0.4  0.01       0.05       1.00       0.6                50    
    ##   2          0.4  0.01       0.05       1.00       0.6               100    
    ##   2          0.4  0.01       0.05       1.00       0.6               150    
    ##   2          0.4  0.01       0.05       1.00       0.8                50    
    ##   2          0.4  0.01       0.05       1.00       0.8               100    
    ##   2          0.4  0.01       0.05       1.00       0.8               150    
    ##   2          0.4  0.01       0.95       0.50       0.6                50    
    ##   2          0.4  0.01       0.95       0.50       0.6               100    
    ##   2          0.4  0.01       0.95       0.50       0.6               150    
    ##   2          0.4  0.01       0.95       0.50       0.8                50    
    ##   2          0.4  0.01       0.95       0.50       0.8               100    
    ##   2          0.4  0.01       0.95       0.50       0.8               150    
    ##   2          0.4  0.01       0.95       0.75       0.6                50    
    ##   2          0.4  0.01       0.95       0.75       0.6               100    
    ##   2          0.4  0.01       0.95       0.75       0.6               150    
    ##   2          0.4  0.01       0.95       0.75       0.8                50    
    ##   2          0.4  0.01       0.95       0.75       0.8               100    
    ##   2          0.4  0.01       0.95       0.75       0.8               150    
    ##   2          0.4  0.01       0.95       1.00       0.6                50    
    ##   2          0.4  0.01       0.95       1.00       0.6               100    
    ##   2          0.4  0.01       0.95       1.00       0.6               150    
    ##   2          0.4  0.01       0.95       1.00       0.8                50    
    ##   2          0.4  0.01       0.95       1.00       0.8               100    
    ##   2          0.4  0.01       0.95       1.00       0.8               150    
    ##   2          0.4  0.50       0.05       0.50       0.6                50    
    ##   2          0.4  0.50       0.05       0.50       0.6               100    
    ##   2          0.4  0.50       0.05       0.50       0.6               150    
    ##   2          0.4  0.50       0.05       0.50       0.8                50    
    ##   2          0.4  0.50       0.05       0.50       0.8               100    
    ##   2          0.4  0.50       0.05       0.50       0.8               150    
    ##   2          0.4  0.50       0.05       0.75       0.6                50    
    ##   2          0.4  0.50       0.05       0.75       0.6               100    
    ##   2          0.4  0.50       0.05       0.75       0.6               150    
    ##   2          0.4  0.50       0.05       0.75       0.8                50    
    ##   2          0.4  0.50       0.05       0.75       0.8               100    
    ##   2          0.4  0.50       0.05       0.75       0.8               150    
    ##   2          0.4  0.50       0.05       1.00       0.6                50    
    ##   2          0.4  0.50       0.05       1.00       0.6               100    
    ##   2          0.4  0.50       0.05       1.00       0.6               150    
    ##   2          0.4  0.50       0.05       1.00       0.8                50    
    ##   2          0.4  0.50       0.05       1.00       0.8               100    
    ##   2          0.4  0.50       0.05       1.00       0.8               150    
    ##   2          0.4  0.50       0.95       0.50       0.6                50    
    ##   2          0.4  0.50       0.95       0.50       0.6               100    
    ##   2          0.4  0.50       0.95       0.50       0.6               150    
    ##   2          0.4  0.50       0.95       0.50       0.8                50    
    ##   2          0.4  0.50       0.95       0.50       0.8               100    
    ##   2          0.4  0.50       0.95       0.50       0.8               150    
    ##   2          0.4  0.50       0.95       0.75       0.6                50    
    ##   2          0.4  0.50       0.95       0.75       0.6               100    
    ##   2          0.4  0.50       0.95       0.75       0.6               150    
    ##   2          0.4  0.50       0.95       0.75       0.8                50    
    ##   2          0.4  0.50       0.95       0.75       0.8               100    
    ##   2          0.4  0.50       0.95       0.75       0.8               150    
    ##   2          0.4  0.50       0.95       1.00       0.6                50    
    ##   2          0.4  0.50       0.95       1.00       0.6               100    
    ##   2          0.4  0.50       0.95       1.00       0.6               150    
    ##   2          0.4  0.50       0.95       1.00       0.8                50    
    ##   2          0.4  0.50       0.95       1.00       0.8               100    
    ##   2          0.4  0.50       0.95       1.00       0.8               150    
    ##   3          0.3  0.01       0.05       0.50       0.6                50    
    ##   3          0.3  0.01       0.05       0.50       0.6               100    
    ##   3          0.3  0.01       0.05       0.50       0.6               150    
    ##   3          0.3  0.01       0.05       0.50       0.8                50    
    ##   3          0.3  0.01       0.05       0.50       0.8               100    
    ##   3          0.3  0.01       0.05       0.50       0.8               150    
    ##   3          0.3  0.01       0.05       0.75       0.6                50    
    ##   3          0.3  0.01       0.05       0.75       0.6               100    
    ##   3          0.3  0.01       0.05       0.75       0.6               150    
    ##   3          0.3  0.01       0.05       0.75       0.8                50    
    ##   3          0.3  0.01       0.05       0.75       0.8               100    
    ##   3          0.3  0.01       0.05       0.75       0.8               150    
    ##   3          0.3  0.01       0.05       1.00       0.6                50    
    ##   3          0.3  0.01       0.05       1.00       0.6               100    
    ##   3          0.3  0.01       0.05       1.00       0.6               150    
    ##   3          0.3  0.01       0.05       1.00       0.8                50    
    ##   3          0.3  0.01       0.05       1.00       0.8               100    
    ##   3          0.3  0.01       0.05       1.00       0.8               150    
    ##   3          0.3  0.01       0.95       0.50       0.6                50    
    ##   3          0.3  0.01       0.95       0.50       0.6               100    
    ##   3          0.3  0.01       0.95       0.50       0.6               150    
    ##   3          0.3  0.01       0.95       0.50       0.8                50    
    ##   3          0.3  0.01       0.95       0.50       0.8               100    
    ##   3          0.3  0.01       0.95       0.50       0.8               150    
    ##   3          0.3  0.01       0.95       0.75       0.6                50    
    ##   3          0.3  0.01       0.95       0.75       0.6               100    
    ##   3          0.3  0.01       0.95       0.75       0.6               150    
    ##   3          0.3  0.01       0.95       0.75       0.8                50    
    ##   3          0.3  0.01       0.95       0.75       0.8               100    
    ##   3          0.3  0.01       0.95       0.75       0.8               150    
    ##   3          0.3  0.01       0.95       1.00       0.6                50    
    ##   3          0.3  0.01       0.95       1.00       0.6               100    
    ##   3          0.3  0.01       0.95       1.00       0.6               150    
    ##   3          0.3  0.01       0.95       1.00       0.8                50    
    ##   3          0.3  0.01       0.95       1.00       0.8               100    
    ##   3          0.3  0.01       0.95       1.00       0.8               150    
    ##   3          0.3  0.50       0.05       0.50       0.6                50    
    ##   3          0.3  0.50       0.05       0.50       0.6               100    
    ##   3          0.3  0.50       0.05       0.50       0.6               150    
    ##   3          0.3  0.50       0.05       0.50       0.8                50    
    ##   3          0.3  0.50       0.05       0.50       0.8               100    
    ##   3          0.3  0.50       0.05       0.50       0.8               150    
    ##   3          0.3  0.50       0.05       0.75       0.6                50    
    ##   3          0.3  0.50       0.05       0.75       0.6               100    
    ##   3          0.3  0.50       0.05       0.75       0.6               150    
    ##   3          0.3  0.50       0.05       0.75       0.8                50    
    ##   3          0.3  0.50       0.05       0.75       0.8               100    
    ##   3          0.3  0.50       0.05       0.75       0.8               150    
    ##   3          0.3  0.50       0.05       1.00       0.6                50    
    ##   3          0.3  0.50       0.05       1.00       0.6               100    
    ##   3          0.3  0.50       0.05       1.00       0.6               150    
    ##   3          0.3  0.50       0.05       1.00       0.8                50    
    ##   3          0.3  0.50       0.05       1.00       0.8               100    
    ##   3          0.3  0.50       0.05       1.00       0.8               150    
    ##   3          0.3  0.50       0.95       0.50       0.6                50    
    ##   3          0.3  0.50       0.95       0.50       0.6               100    
    ##   3          0.3  0.50       0.95       0.50       0.6               150    
    ##   3          0.3  0.50       0.95       0.50       0.8                50    
    ##   3          0.3  0.50       0.95       0.50       0.8               100    
    ##   3          0.3  0.50       0.95       0.50       0.8               150    
    ##   3          0.3  0.50       0.95       0.75       0.6                50    
    ##   3          0.3  0.50       0.95       0.75       0.6               100    
    ##   3          0.3  0.50       0.95       0.75       0.6               150    
    ##   3          0.3  0.50       0.95       0.75       0.8                50    
    ##   3          0.3  0.50       0.95       0.75       0.8               100    
    ##   3          0.3  0.50       0.95       0.75       0.8               150    
    ##   3          0.3  0.50       0.95       1.00       0.6                50    
    ##   3          0.3  0.50       0.95       1.00       0.6               100    
    ##   3          0.3  0.50       0.95       1.00       0.6               150    
    ##   3          0.3  0.50       0.95       1.00       0.8                50    
    ##   3          0.3  0.50       0.95       1.00       0.8               100    
    ##   3          0.3  0.50       0.95       1.00       0.8               150    
    ##   3          0.4  0.01       0.05       0.50       0.6                50    
    ##   3          0.4  0.01       0.05       0.50       0.6               100    
    ##   3          0.4  0.01       0.05       0.50       0.6               150    
    ##   3          0.4  0.01       0.05       0.50       0.8                50    
    ##   3          0.4  0.01       0.05       0.50       0.8               100    
    ##   3          0.4  0.01       0.05       0.50       0.8               150    
    ##   3          0.4  0.01       0.05       0.75       0.6                50    
    ##   3          0.4  0.01       0.05       0.75       0.6               100    
    ##   3          0.4  0.01       0.05       0.75       0.6               150    
    ##   3          0.4  0.01       0.05       0.75       0.8                50    
    ##   3          0.4  0.01       0.05       0.75       0.8               100    
    ##   3          0.4  0.01       0.05       0.75       0.8               150    
    ##   3          0.4  0.01       0.05       1.00       0.6                50    
    ##   3          0.4  0.01       0.05       1.00       0.6               100    
    ##   3          0.4  0.01       0.05       1.00       0.6               150    
    ##   3          0.4  0.01       0.05       1.00       0.8                50    
    ##   3          0.4  0.01       0.05       1.00       0.8               100    
    ##   3          0.4  0.01       0.05       1.00       0.8               150    
    ##   3          0.4  0.01       0.95       0.50       0.6                50    
    ##   3          0.4  0.01       0.95       0.50       0.6               100    
    ##   3          0.4  0.01       0.95       0.50       0.6               150    
    ##   3          0.4  0.01       0.95       0.50       0.8                50    
    ##   3          0.4  0.01       0.95       0.50       0.8               100    
    ##   3          0.4  0.01       0.95       0.50       0.8               150    
    ##   3          0.4  0.01       0.95       0.75       0.6                50    
    ##   3          0.4  0.01       0.95       0.75       0.6               100    
    ##   3          0.4  0.01       0.95       0.75       0.6               150    
    ##   3          0.4  0.01       0.95       0.75       0.8                50    
    ##   3          0.4  0.01       0.95       0.75       0.8               100    
    ##   3          0.4  0.01       0.95       0.75       0.8               150    
    ##   3          0.4  0.01       0.95       1.00       0.6                50    
    ##   3          0.4  0.01       0.95       1.00       0.6               100    
    ##   3          0.4  0.01       0.95       1.00       0.6               150    
    ##   3          0.4  0.01       0.95       1.00       0.8                50    
    ##   3          0.4  0.01       0.95       1.00       0.8               100    
    ##   3          0.4  0.01       0.95       1.00       0.8               150    
    ##   3          0.4  0.50       0.05       0.50       0.6                50    
    ##   3          0.4  0.50       0.05       0.50       0.6               100    
    ##   3          0.4  0.50       0.05       0.50       0.6               150    
    ##   3          0.4  0.50       0.05       0.50       0.8                50    
    ##   3          0.4  0.50       0.05       0.50       0.8               100    
    ##   3          0.4  0.50       0.05       0.50       0.8               150    
    ##   3          0.4  0.50       0.05       0.75       0.6                50    
    ##   3          0.4  0.50       0.05       0.75       0.6               100    
    ##   3          0.4  0.50       0.05       0.75       0.6               150    
    ##   3          0.4  0.50       0.05       0.75       0.8                50    
    ##   3          0.4  0.50       0.05       0.75       0.8               100    
    ##   3          0.4  0.50       0.05       0.75       0.8               150    
    ##   3          0.4  0.50       0.05       1.00       0.6                50    
    ##   3          0.4  0.50       0.05       1.00       0.6               100    
    ##   3          0.4  0.50       0.05       1.00       0.6               150    
    ##   3          0.4  0.50       0.05       1.00       0.8                50    
    ##   3          0.4  0.50       0.05       1.00       0.8               100    
    ##   3          0.4  0.50       0.05       1.00       0.8               150    
    ##   3          0.4  0.50       0.95       0.50       0.6                50    
    ##   3          0.4  0.50       0.95       0.50       0.6               100    
    ##   3          0.4  0.50       0.95       0.50       0.6               150    
    ##   3          0.4  0.50       0.95       0.50       0.8                50    
    ##   3          0.4  0.50       0.95       0.50       0.8               100    
    ##   3          0.4  0.50       0.95       0.50       0.8               150    
    ##   3          0.4  0.50       0.95       0.75       0.6                50    
    ##   3          0.4  0.50       0.95       0.75       0.6               100    
    ##   3          0.4  0.50       0.95       0.75       0.6               150    
    ##   3          0.4  0.50       0.95       0.75       0.8                50    
    ##   3          0.4  0.50       0.95       0.75       0.8               100    
    ##   3          0.4  0.50       0.95       0.75       0.8               150    
    ##   3          0.4  0.50       0.95       1.00       0.6                50    
    ##   3          0.4  0.50       0.95       1.00       0.6               100    
    ##   3          0.4  0.50       0.95       1.00       0.6               150    
    ##   3          0.4  0.50       0.95       1.00       0.8                50    
    ##   3          0.4  0.50       0.95       1.00       0.8               100    
    ##   3          0.4  0.50       0.95       1.00       0.8               150    
    ##   RMSE      Rsquared   MAE     
    ##   2.997138  0.8611539  2.488265
    ##   2.418221  0.8687395  1.956238
    ##   1.718277  0.8758219  1.230636
    ##   3.632041  0.8520026  3.237118
    ##   2.508958  0.8454030  1.941823
    ##   1.902259  0.8547664  1.416964
    ##   3.151540  0.8493065  2.719438
    ##   2.233191  0.8553323  1.785870
    ##   1.847357  0.8600253  1.328205
    ##   2.993514  0.9107104  2.597638
    ##   2.195507  0.9047362  1.776282
    ##   1.602500  0.9037952  1.181297
    ##   2.790876  0.8896705  2.353836
    ##   2.178239  0.8992909  1.711167
    ##   1.667953  0.8996696  1.208543
    ##   3.271711  0.8841017  2.944500
    ##   2.116566  0.8853251  1.648664
    ##   1.582318  0.8899727  1.142259
    ##   2.023971  0.8271922  1.477806
    ##   1.887602  0.8425341  1.348985
    ##   1.871344  0.8348914  1.348048
    ##   1.660479  0.8908483  1.138652
    ##   1.707105  0.8912853  1.190891
    ##   1.707240  0.8933817  1.197273
    ##   1.729583  0.8740170  1.271825
    ##   1.676658  0.8790543  1.261354
    ##   1.683497  0.8786427  1.288588
    ##   1.679067  0.8846675  1.269084
    ##   1.674905  0.8883737  1.287142
    ##   1.659961  0.8906026  1.274939
    ##   1.613604  0.8989380  1.123433
    ##   1.588011  0.8986575  1.138546
    ##   1.607148  0.8991876  1.161767
    ##   1.598560  0.9077581  1.149791
    ##   1.589058  0.9076930  1.138479
    ##   1.572397  0.9086464  1.128808
    ##   6.299777  0.8325799  5.711136
    ##   3.253848  0.8479898  2.580726
    ##   1.949968  0.8549718  1.419715
    ##   6.745003  0.8456551  6.120141
    ##   3.622852  0.8356917  2.877242
    ##   1.914010  0.8669138  1.363604
    ##   6.035975  0.8730914  5.545755
    ##   2.978139  0.8895555  2.473568
    ##   1.746165  0.8905839  1.267428
    ##   6.288613  0.8560674  5.795963
    ##   3.385080  0.8619523  2.745810
    ##   1.957631  0.8653557  1.443105
    ##   6.020322  0.8743613  5.555741
    ##   3.141024  0.8781087  2.647832
    ##   1.942675  0.8822427  1.349874
    ##   6.118919  0.8823426  5.702126
    ##   3.008960  0.8841715  2.515150
    ##   1.777204  0.8898278  1.237474
    ##   1.915151  0.8573901  1.368618
    ##   1.864291  0.8653256  1.318474
    ##   1.834093  0.8652862  1.323822
    ##   1.769922  0.8942744  1.285937
    ##   1.731362  0.9064515  1.264800
    ##   1.704756  0.9057329  1.204477
    ##   1.707372  0.8645309  1.381766
    ##   1.692271  0.8612971  1.348041
    ##   1.667516  0.8617394  1.340923
    ##   1.775918  0.8962135  1.351733
    ##   1.825328  0.8900906  1.360915
    ##   1.821699  0.8915966  1.352189
    ##   1.643364  0.9019954  1.192025
    ##   1.587263  0.9037622  1.139975
    ##   1.572488  0.9061705  1.141045
    ##   1.646502  0.9046646  1.212907
    ##   1.611573  0.9064276  1.179852
    ##   1.584951  0.9079712  1.164678
    ##   4.256580  0.8340940  3.737350
    ##   2.781608  0.8244246  2.081668
    ##   2.021457  0.8303946  1.506297
    ##   4.312798  0.8546883  3.861649
    ##   2.662135  0.8679662  2.246653
    ##   1.723004  0.8741853  1.392702
    ##   4.209904  0.8580814  3.751610
    ##   3.149370  0.8701317  2.691162
    ##   1.871416  0.8667998  1.371959
    ##   3.629114  0.8667554  3.218162
    ##   2.568454  0.8719301  2.190324
    ##   1.896182  0.8725446  1.347733
    ##   3.821561  0.9183745  3.491450
    ##   2.381647  0.9134469  1.926033
    ##   1.574245  0.9144519  1.136881
    ##   3.444161  0.9106763  3.048931
    ##   2.308261  0.9139215  1.946986
    ##   1.546619  0.9093158  1.154491
    ##   2.113065  0.8587372  1.576197
    ##   2.016930  0.8657076  1.494529
    ##   1.926844  0.8581197  1.407604
    ##   1.896808  0.8551924  1.365064
    ##   1.935158  0.8622683  1.394371
    ##   1.883186  0.8544801  1.356218
    ##   1.575499  0.9096423  1.197811
    ##   1.578389  0.9082416  1.234723
    ##   1.584905  0.9057508  1.236630
    ##   1.796613  0.8698971  1.226359
    ##   1.855817  0.8609003  1.294167
    ##   1.855764  0.8577102  1.300866
    ##   1.686735  0.9146978  1.236822
    ##   1.705416  0.9072812  1.280838
    ##   1.673080  0.9065933  1.237092
    ##   1.626061  0.9049679  1.225662
    ##   1.587475  0.9079439  1.190933
    ##   1.564634  0.9100033  1.163648
    ##   7.259091  0.8572545  6.739689
    ##   3.571062  0.8758651  2.993354
    ##   1.785374  0.8913362  1.356086
    ##   7.430053  0.8046001  6.865153
    ##   3.815164  0.7982785  3.149213
    ##   2.026799  0.8252746  1.430989
    ##   7.075507  0.8631198  6.622448
    ##   3.569042  0.8529667  3.017961
    ##   1.967494  0.8550190  1.407002
    ##   7.249947  0.8696823  6.750528
    ##   3.588336  0.8751634  3.049790
    ##   1.829258  0.8802690  1.293256
    ##   6.906970  0.8694580  6.510002
    ##   3.334964  0.8793187  2.887369
    ##   1.779000  0.8870504  1.212579
    ##   7.193871  0.8807273  6.783973
    ##   3.463331  0.8865269  2.990851
    ##   1.786191  0.8938769  1.272886
    ##   1.798596  0.8492489  1.406551
    ##   1.684312  0.8632049  1.350035
    ##   1.686649  0.8629045  1.342197
    ##   1.957516  0.8492660  1.413065
    ##   1.818566  0.8583218  1.249122
    ##   1.825590  0.8569645  1.242948
    ##   1.822345  0.8599004  1.330915
    ##   1.738756  0.8593523  1.301407
    ##   1.713869  0.8591291  1.309928
    ##   1.921518  0.8796920  1.476821
    ##   1.754371  0.8781075  1.335608
    ##   1.750619  0.8762118  1.340212
    ##   1.748144  0.9045453  1.241416
    ##   1.682491  0.9038457  1.194716
    ##   1.661003  0.9034378  1.205185
    ##   1.686613  0.9029352  1.228938
    ##   1.600870  0.9061581  1.135667
    ##   1.567157  0.9042732  1.096114
    ##   3.050512  0.8338727  2.463744
    ##   2.265607  0.8360567  1.777649
    ##   1.871851  0.8396229  1.468463
    ##   2.582857  0.8501939  2.089889
    ##   2.088092  0.8549869  1.575045
    ##   1.923512  0.8589137  1.442534
    ##   3.034963  0.8996494  2.682035
    ##   2.208775  0.9003643  1.779854
    ##   1.676816  0.8999453  1.201033
    ##   3.586812  0.8529950  3.115467
    ##   2.537732  0.8494526  2.081040
    ##   1.976455  0.8526067  1.460296
    ##   2.866372  0.8857880  2.432986
    ##   1.972245  0.8850550  1.502981
    ##   1.766046  0.8851033  1.251344
    ##   3.714782  0.8665901  3.259741
    ##   2.055144  0.8708248  1.594391
    ##   1.788386  0.8734805  1.252613
    ##   2.027208  0.8498137  1.457162
    ##   2.031528  0.8539246  1.469121
    ##   1.999987  0.8534657  1.442861
    ##   1.884695  0.8393722  1.418592
    ##   1.823215  0.8302670  1.392611
    ##   1.841377  0.8273821  1.425281
    ##   1.693759  0.8709270  1.253141
    ##   1.718932  0.8678103  1.292603
    ##   1.772340  0.8691401  1.313860
    ##   1.904856  0.8562932  1.364951
    ##   1.879019  0.8540472  1.292307
    ##   1.858409  0.8544860  1.284755
    ##   1.802689  0.8780025  1.280491
    ##   1.791335  0.8762463  1.288187
    ##   1.803067  0.8757076  1.300940
    ##   1.731303  0.8754097  1.261556
    ##   1.720291  0.8792786  1.264318
    ##   1.718963  0.8792732  1.265632
    ##   6.687430  0.8486808  6.042999
    ##   3.601039  0.8552574  2.927366
    ##   2.089352  0.8579101  1.417095
    ##   6.535666  0.8634189  5.958328
    ##   3.269783  0.8591804  2.688801
    ##   2.029304  0.8612375  1.419993
    ##   6.222230  0.8444367  5.697211
    ##   3.219992  0.8596840  2.715199
    ##   1.830014  0.8698082  1.267049
    ##   6.247903  0.8439610  5.720638
    ##   3.154731  0.8646838  2.639123
    ##   1.890469  0.8727244  1.377685
    ##   6.286962  0.8609085  5.836340
    ##   3.198390  0.8712909  2.697142
    ##   1.786275  0.8776745  1.244450
    ##   6.171636  0.8862392  5.767598
    ##   3.007662  0.8935051  2.584550
    ##   1.783190  0.8917320  1.283514
    ##   2.045794  0.8166415  1.484690
    ##   2.018200  0.8138429  1.483893
    ##   2.026021  0.8113143  1.496498
    ##   1.869387  0.8264903  1.326314
    ##   1.879201  0.8299348  1.355116
    ##   1.879431  0.8283212  1.376031
    ##   1.701916  0.8977804  1.226127
    ##   1.711576  0.8931295  1.257197
    ##   1.683377  0.8907301  1.244953
    ##   1.783060  0.8704808  1.311638
    ##   1.739509  0.8687146  1.277426
    ##   1.767016  0.8692295  1.314519
    ##   1.704785  0.8982393  1.239567
    ##   1.704117  0.8931836  1.239427
    ##   1.710473  0.8922946  1.252673
    ##   1.690191  0.8912245  1.186648
    ##   1.647361  0.8912329  1.148217
    ##   1.662947  0.8909342  1.166047
    ##   3.526523  0.8494373  3.098446
    ##   2.217311  0.8494306  1.653143
    ##   1.891633  0.8571379  1.307242
    ##   4.116967  0.8612159  3.688681
    ##   2.420436  0.8579185  1.897984
    ##   1.879789  0.8578733  1.421077
    ##   4.713236  0.8717038  4.350378
    ##   3.014524  0.8670543  2.629740
    ##   1.865843  0.8723119  1.301643
    ##   4.524351  0.8946441  4.158934
    ##   2.813223  0.8876445  2.386481
    ##   1.735771  0.8899464  1.153722
    ##   2.660374  0.8708003  2.300264
    ##   2.140683  0.8685072  1.709569
    ##   1.807525  0.8675993  1.329620
    ##   3.652219  0.8897775  3.296236
    ##   2.429451  0.8876659  2.059897
    ##   1.784963  0.8942459  1.292306
    ##   2.216440  0.8085868  1.539431
    ##   2.208482  0.8089465  1.519004
    ##   2.214849  0.8086043  1.522492
    ##   1.954167  0.8661874  1.478135
    ##   2.002815  0.8609168  1.523775
    ##   1.961290  0.8617214  1.510507
    ##   1.833922  0.8808363  1.317970
    ##   1.812771  0.8801445  1.296777
    ##   1.812106  0.8797914  1.291336
    ##   1.894795  0.8686754  1.360991
    ##   1.894131  0.8689560  1.376188
    ##   1.889255  0.8698633  1.374341
    ##   1.780708  0.8844497  1.194637
    ##   1.769887  0.8858644  1.190971
    ##   1.766245  0.8862333  1.188704
    ##   1.799199  0.8760870  1.262836
    ##   1.790813  0.8789861  1.271612
    ##   1.733194  0.8793159  1.207300
    ##   7.467502  0.8681643  6.907074
    ##   3.794461  0.8774740  3.173662
    ##   1.936193  0.8570118  1.358293
    ##   7.117095  0.8500683  6.644477
    ##   3.713808  0.8746338  3.147292
    ##   1.935631  0.8797894  1.393251
    ##   7.383292  0.8621330  6.870986
    ##   3.656617  0.8759171  3.127160
    ##   1.833686  0.8813243  1.274011
    ##   7.249370  0.8685250  6.782287
    ##   3.496472  0.8775033  2.998513
    ##   1.759666  0.8815332  1.217287
    ##   7.003594  0.8722115  6.592375
    ##   3.457759  0.8706106  3.007762
    ##   1.760069  0.8806803  1.223694
    ##   7.305994  0.8828072  6.896343
    ##   3.569257  0.8962065  3.113783
    ##   1.716586  0.8995419  1.240084
    ##   1.760840  0.8941730  1.310345
    ##   1.692986  0.8915059  1.304155
    ##   1.653123  0.8914814  1.297741
    ##   1.722189  0.8581804  1.312632
    ##   1.720778  0.8612799  1.281790
    ##   1.715892  0.8603878  1.301020
    ##   1.854027  0.8880800  1.349533
    ##   1.800996  0.8879168  1.293815
    ##   1.780514  0.8873026  1.276282
    ##   2.219094  0.8611843  1.673936
    ##   2.198179  0.8652705  1.590697
    ##   2.199216  0.8665741  1.585328
    ##   1.777891  0.8973343  1.313020
    ##   1.742081  0.8969657  1.286452
    ##   1.743743  0.8978771  1.293416
    ##   1.760225  0.8872355  1.211741
    ##   1.695846  0.8872871  1.181871
    ##   1.672314  0.8871636  1.177014
    ##   3.575170  0.8164075  2.994167
    ##   2.554761  0.8228017  1.935980
    ##   2.096477  0.8244483  1.520520
    ##   3.268246  0.8776418  2.808644
    ##   2.009310  0.8778050  1.514385
    ##   1.785009  0.8735025  1.270955
    ##   3.179137  0.8824214  2.784231
    ##   2.175325  0.8809686  1.719717
    ##   1.763164  0.8772527  1.380701
    ##   3.216752  0.8662144  2.799781
    ##   2.208887  0.8708888  1.731018
    ##   1.720835  0.8746187  1.289013
    ##   2.664353  0.8806062  2.146832
    ##   2.021643  0.8804838  1.573374
    ##   1.811199  0.8823342  1.346909
    ##   4.239642  0.9098750  3.886886
    ##   2.236808  0.9023789  1.832854
    ##   1.563768  0.9040123  1.166037
    ##   1.759108  0.8816066  1.310468
    ##   1.735613  0.8781972  1.306266
    ##   1.740752  0.8771581  1.307272
    ##   2.158537  0.8254691  1.569052
    ##   2.113032  0.8265242  1.555586
    ##   2.068071  0.8257620  1.509900
    ##   1.919463  0.8504348  1.357549
    ##   1.947843  0.8484294  1.377233
    ##   1.933355  0.8477847  1.373276
    ##   1.913544  0.8627520  1.347558
    ##   1.931293  0.8616415  1.360069
    ##   1.896684  0.8612053  1.331011
    ##   1.659676  0.8990026  1.137568
    ##   1.623196  0.8983849  1.130676
    ##   1.621811  0.8980100  1.131823
    ##   1.634831  0.8941394  1.214457
    ##   1.666542  0.8934594  1.195230
    ##   1.681836  0.8938396  1.210648
    ##   6.350052  0.8510715  5.800497
    ##   3.425343  0.8585396  2.774058
    ##   1.901537  0.8837000  1.299130
    ##   6.702110  0.8302183  6.057770
    ##   3.428123  0.8568344  2.772721
    ##   1.957002  0.8766476  1.401160
    ##   6.298839  0.8834309  5.796543
    ##   3.136172  0.8881128  2.629059
    ##   1.835103  0.8937060  1.306375
    ##   6.276958  0.8727211  5.764249
    ##   3.366203  0.8714999  2.776453
    ##   2.036169  0.8757409  1.443740
    ##   6.282640  0.8659436  5.844587
    ##   3.212584  0.8880993  2.761621
    ##   1.753872  0.8870763  1.217637
    ##   6.156449  0.8842449  5.726733
    ##   2.941177  0.8996159  2.495531
    ##   1.751390  0.8963837  1.242596
    ##   1.842202  0.8636725  1.314756
    ##   1.849404  0.8600068  1.324697
    ##   1.840428  0.8599833  1.320215
    ##   1.855119  0.8310584  1.358317
    ##   1.838834  0.8355712  1.368126
    ##   1.837010  0.8358121  1.363687
    ##   1.784249  0.8642467  1.188179
    ##   1.721730  0.8663118  1.159408
    ##   1.730721  0.8662690  1.169928
    ##   1.836368  0.8658792  1.319990
    ##   1.783537  0.8673410  1.273247
    ##   1.776732  0.8678250  1.261690
    ##   1.574247  0.8996707  1.137338
    ##   1.556948  0.8986973  1.102859
    ##   1.571444  0.8986505  1.108912
    ##   1.674564  0.8987672  1.196104
    ##   1.622698  0.9003237  1.145499
    ##   1.631877  0.9004597  1.153804
    ##   2.311600  0.8625189  1.812591
    ##   1.968434  0.8645015  1.455158
    ##   1.628495  0.8685175  1.186005
    ##   3.304570  0.8533875  2.963258
    ##   2.247061  0.8601470  1.809060
    ##   1.695850  0.8635523  1.223340
    ##   3.606703  0.8321869  3.164192
    ##   2.445702  0.8361106  1.986871
    ##   1.974363  0.8386331  1.370397
    ##   4.449089  0.8871578  4.200933
    ##   2.627222  0.8841909  2.134160
    ##   1.795949  0.8858398  1.251256
    ##   4.003810  0.8351764  3.625944
    ##   2.462815  0.8334873  1.965086
    ##   1.858610  0.8377718  1.207701
    ##   3.839516  0.8610527  3.415994
    ##   3.205361  0.8599759  2.765514
    ##   1.833887  0.8672573  1.265452
    ##   1.870390  0.8825030  1.447825
    ##   1.709296  0.8846591  1.339134
    ##   1.705600  0.8845343  1.341291
    ##   2.126353  0.8031047  1.495510
    ##   2.109580  0.8039416  1.485785
    ##   2.054858  0.8042163  1.473925
    ##   1.573814  0.8904301  1.084514
    ##   1.573133  0.8905834  1.081511
    ##   1.591436  0.8905554  1.091616
    ##   1.953971  0.8515671  1.353721
    ##   1.958164  0.8512422  1.355758
    ##   2.014426  0.8503280  1.390186
    ##   1.949420  0.8725251  1.439008
    ##   1.768878  0.8690262  1.235638
    ##   1.762435  0.8690509  1.231764
    ##   1.728682  0.9006897  1.279977
    ##   1.630104  0.9006139  1.175360
    ##   1.566945  0.9004967  1.124132
    ##   7.298770  0.8833622  6.787416
    ##   3.733020  0.8913821  3.148417
    ##   1.937564  0.8838167  1.340315
    ##   7.191989  0.8656567  6.669126
    ##   3.500661  0.8508044  2.951953
    ##   1.916351  0.8573115  1.307254
    ##   7.356251  0.8957262  6.897225
    ##   3.674302  0.8851928  3.142363
    ##   1.769188  0.8852878  1.165224
    ##   7.356529  0.8410622  6.864673
    ##   3.709113  0.8755517  3.155325
    ##   1.750597  0.8802990  1.191136
    ##   6.996806  0.8854839  6.597691
    ##   3.330594  0.8940466  2.903456
    ##   1.739250  0.8975288  1.186037
    ##   7.136588  0.8784283  6.735079
    ##   3.451017  0.8871335  2.987516
    ##   1.751409  0.8888925  1.199247
    ##   2.014561  0.8581130  1.509373
    ##   1.988793  0.8589717  1.436396
    ##   1.984249  0.8599856  1.427114
    ##   1.836063  0.8855502  1.367636
    ##   1.748037  0.8887457  1.307887
    ##   1.737305  0.8890171  1.306347
    ##   1.961497  0.8648320  1.369742
    ##   1.914800  0.8654806  1.329957
    ##   1.924858  0.8660296  1.333950
    ##   2.008048  0.8606985  1.504936
    ##   1.956206  0.8604022  1.443033
    ##   1.913715  0.8608227  1.373559
    ##   1.868907  0.8578333  1.367178
    ##   1.879921  0.8587130  1.358379
    ##   1.848294  0.8586934  1.354846
    ##   1.719903  0.8873610  1.215595
    ##   1.688703  0.8885114  1.172788
    ##   1.676094  0.8887217  1.161336
    ## 
    ## Tuning parameter 'gamma' was held constant at a value of 0
    ## Tuning
    ##  parameter 'min_child_weight' was held constant at a value of 1
    ## Rsquared was used to select the optimal model using the largest value.
    ## The final values used for the model were nrounds = 50, max_depth = 1, eta
    ##  = 0.4, gamma = 0, subsample = 1, colsample_bytree = 0.6, rate_drop =
    ##  0.01, skip_drop = 0.05 and min_child_weight = 1.

## 8.10 Evaluating models

``` r
predict_models_2 <- predict(model_trained_2, newdata = test)

for (i in 1:length(predict_models_2)){
  
  print(names(predict_models_2[i]))
  
  result <- postResample(pred = predict_models_2[[i]], obs = test$Consumption)
  
  print(result)
  
  print("------------------------------------------------------------------------------")
}
```

    ## [1] "lm"
    ##      RMSE  Rsquared       MAE 
    ## 1.2066970 0.9409441 0.8921191 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "ridge"
    ##      RMSE  Rsquared       MAE 
    ## 1.2066970 0.9409441 0.8921191 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "rf"
    ##      RMSE  Rsquared       MAE 
    ## 1.4918101 0.9156421 1.2412446 
    ## [1] "------------------------------------------------------------------------------"
    ## [1] "xgbDART"
    ##      RMSE  Rsquared       MAE 
    ## 1.6437143 0.8864712 1.3162674 
    ## [1] "------------------------------------------------------------------------------"

Based on the results obtained, and to be consistent with what we
previously determined, we decided to choose **Linear Regression** as our
prediction model.

## 8.11 Ranking features by importance

``` r
chosen_model_2 <- train(`Consumption` ~ ., data = train_validation, 
                        method = "lm",
                        trControl = train.control,
                        metric = 'Rsquared')

importance <- varImp(chosen_model_2, scale=TRUE)

plot(importance)
```

![](electric_cars_prediction_files/figure-gfm/unnamed-chunk-69-1.png)<!-- -->

``` r
summary(chosen_model_2)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##      Min       1Q   Median       3Q      Max 
    ## -2.49198 -0.29459  0.09462  0.47830  2.06641 
    ## 
    ## Coefficients:
    ##                  Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)      18.87235    0.16192 116.551  < 2e-16 ***
    ## Power             3.81948    1.02216   3.737 0.000783 ***
    ## Torque            1.15814    0.74226   1.560 0.129180    
    ## Battery_Capacity  0.10913    0.61063   0.179 0.859362    
    ## Gross_Weight      0.84931    0.78124   1.087 0.285642    
    ## Load_Capacity     1.47724    0.34026   4.341 0.000148 ***
    ## Tire_Size        -1.93495    0.40456  -4.783  4.3e-05 ***
    ## Max_Speed        -2.52586    0.58364  -4.328 0.000154 ***
    ## Boot_Capacity    -0.03309    0.43582  -0.076 0.939990    
    ## Drive_Type_2WD   -1.19778    0.35757  -3.350 0.002195 ** 
    ## Drive_Type_4WD    0.55584    0.45334   1.226 0.229697    
    ## Type_Brakes_disc  0.25315    0.22861   1.107 0.276947    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 1.049 on 30 degrees of freedom
    ## Multiple R-squared:   0.96,  Adjusted R-squared:  0.9453 
    ## F-statistic: 65.38 on 11 and 30 DF,  p-value: < 2.2e-16

We have two features, `Battery_Capacity` and `Boot_Capacity`, with
higher p-values. This way, for our optimized model we decided to exclude
them from our predictor features.

## 8.12 Optmizing model

``` r
opt_model_2 <- train(`Consumption` ~ ., data = train_validation[, -c(3,8)], 
                     method = "lm",
                     trControl = train.control,
                     metric = 'Rsquared')


predict_opt_2 <- predict(opt_model_2, newdata = test)

result_df_2 <- postResample(pred = predict_opt_2, obs = test$Consumption)

result_df_2
```

    ##      RMSE  Rsquared       MAE 
    ## 1.1997729 0.9423740 0.8958421

# 9. Final considerations

- By inputting missing values in our dataset we were able to outperform
  our first prediction model. Our chosen model would be, then, a
  **Linear Regression** with the following metrics: **R² = 0.94**, **MAE
  = 0.89** and **RMSE = 1,20**. Having the mean of `Consumption` from
  our initial dataset as a basis, we obtained a model that achieved
  around 5% of error (in terms of MAE) in its predictions.

- Some extra improvements can be studied, such as the use of other set
  of features, other algorithms and also by making feature engineering
  in the dataset.

- The model is now ready to be deployed to operational usage.
