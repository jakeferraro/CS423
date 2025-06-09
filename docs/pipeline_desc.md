# Bank Marketing Dataset - Data Processing Pipeline

## Dataset Overview
- **Name**: Bank Marketing Dataset
- **Source**: Portuguese bank direct marketing campaigns
- **Target**: Predict term deposit subscription (binary classification)
- **Size**: 11,162 rows × 17 columns (downsampled to 5,000 rows)
- **Target Distribution**: 47.4% positive class (deposit = yes)

## Pipeline Structure

### Stage 1: Binary Categorical Mappings
**Purpose**: Convert simple yes/no categorical variables to binary numeric format

- **map_default**: Maps credit default status (no=0, yes=1)
- **map_housing**: Maps housing loan status (no=0, yes=1)  
- **map_loan**: Maps personal loan status (no=0, yes=1)

### Stage 2: Ordinal Categorical Mappings
**Purpose**: Convert ordinal categories to meaningful numeric scales

- **map_education**: Maps education level with natural ordering
  - unknown=0, primary=1, secondary=2, tertiary=3
- **map_marital**: Maps marital status 
  - single=1, married=2, divorced=3

### Stage 3: Target Encoding
**Purpose**: Handle high-cardinality categorical variables using target-based encoding with smoothing

- **target_job**: Encodes job categories (12 unique values) using target mean with smoothing=10
- **target_month**: Encodes contact month (12 unique values) with smoothing=10
- **target_poutcome**: Encodes previous campaign outcome (4 unique values) with smoothing=10
- **target_contact**: Encodes contact method (3 unique values) with smoothing=10

*Smoothing parameter prevents overfitting on categories with few samples*

### Stage 4: Outlier Detection and Handling
**Purpose**: Identify and cap extreme values using Tukey's method (outer fence)

- **tukey_age**: Handles age outliers
- **tukey_balance**: Handles account balance outliers
- **tukey_duration**: Handles call duration outliers
- **tukey_campaign**: Handles number of contacts outliers
- **tukey_previous**: Handles previous campaign contacts outliers

*Outer fence = Q3 + 3×IQR ensures only extreme outliers are capped*

### Stage 5: Feature Scaling
**Purpose**: Normalize numerical features using robust scaling (less sensitive to outliers)

- **scale_age**: Scales age using median and IQR
- **scale_balance**: Scales account balance 
- **scale_duration**: Scales last contact duration
- **scale_day**: Scales day of month
- **scale_campaign**: Scales number of contacts during campaign
- **scale_pdays**: Scales days since last contact
- **scale_previous**: Scales number of previous contacts

*Robust scaling: (x - median) / IQR*

### Stage 6: Missing Value Imputation
**Purpose**: Handle any remaining missing values

- **impute**: K-Nearest Neighbors imputation with k=5
  - Uses 5 most similar samples to impute missing values
  - Works well with mixed data types after preprocessing
