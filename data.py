import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE


def get_raw_OV_data():
    filepath = "./GEO Samples/Ov_merged_data_with_metadata.csv"
    ov_samples = pd.read_csv(filepath)
    # Drop ID and lable column
    ov = ov_samples.drop(['ID', 'Stage'], axis=1, inplace=False)
    ov_lable = ov_samples[['Stage']]
    print(ov_lable['Stage'].value_counts())

    return ov, ov_lable

def get_raw_non_cancer_data():
    filepath = "./GEO Samples/none_cancer_merged_data_with_metadata.csv"
    none_cancer = pd.read_csv(filepath)
    return none_cancer

def get_ov_over_sampled():
    x, y = get_raw_OV_data()
    X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=20)
    
    # Over Sampling
    smote = BorderlineSMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)    
    y_train_oversampled['Stage'].value_counts()
    
    # Shuffle the oversampled training data
    shuffled = X_train_oversampled.join(y_train_oversampled).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train = shuffled.drop(columns='Stage')
    y_train = shuffled['Stage']

    return X_train, X_test, y_train, y_test 

def geo_100_normalized():
    filepath = "./GEO Samples/train_data_Normalized.csv"
    train_data = pd.read_csv(filepath)
    X_train = train_data.drop(['Stage'], axis=1)
    y_train = train_data[['Stage']]
    
    filepath = "./GEO Samples/test_data_Normalized.csv"
    test_data = pd.read_csv(filepath)
    X_test = test_data.drop(['Stage'], axis=1)
    y_test = test_data[['Stage']]

    return X_train, y_train, X_test, y_test


def get_1060_normalized():
    filepath = "./GEO Samples/all_data_train_data_stage_classification.csv"
    train_data = pd.read_csv(filepath)
    X_train = train_data.drop(['Stage'], axis=1)
    y_train = train_data[['Stage']]
    
    filepath = "./GEO Samples/all_data_test_data_stage_classification.csv.csv"
    test_data = pd.read_csv(filepath)
    X_test = test_data.drop(['Stage'], axis=1)
    y_test = test_data[['Stage']]
    return X_train, y_train, X_test, y_test

def get_Normalized_data_with_43_features():
    filepath = "/GEO Samples/dimension_reduced_data.csv"
    X = pd.read_csv(filepath)        
    filepath = "./GEO Samples/dimension_reduced_data_lables.csv"
    y = pd.read_csv(filepath)
    
    return X, y
    