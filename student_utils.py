import pandas as pd
import numpy as np
import os
import tensorflow as tf

def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
        
        Approach:
        The non-proprietary name lists the active ingredients, which is usually the same even for
        multiple NDC codes that differentiate dosage form, route, etc.
        Therefore, I group by non-proprietary name.
        
    '''
    
    # Grouping fields
    grouping_field_list = ['Non-proprietary Name']
    non_grouped_field_list = [c for c in ndc_df.columns if c not in grouping_field_list]

    grouped_ndc_codes = ndc_df.groupby(grouping_field_list)[non_grouped_field_list].agg(lambda x: 
                                                            list([y for y in x if y is not np.nan ] ) ).reset_index()
    
    reduce_dim_dict = {}
    for val in grouped_ndc_codes['Non-proprietary Name'].unique():
        reduce_dim_dict[val] = grouped_ndc_codes.loc[grouped_ndc_codes['Non-proprietary Name'] == val, 'NDC_Code'].item()

    def findGenericDrugName(ndc_code):
        for key in reduce_dim_dict:
            if (ndc_code in reduce_dim_dict[key]):
                return key
        return ndc_code
        
    df['generic_drug_name'] = df['ndc_code']
    df['generic_drug_name'] = df['generic_drug_name'].map(findGenericDrugName)
    
    return df

def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values(by=['encounter_id']).drop_duplicates(subset="patient_nbr")
    
    return first_encounter_df

# From course example of choosing last encounter: we would use head() instead of tail():
# def select_last_encounter(df, patient_id, encounter_id):
#     df = df.sort_values(encounter_id)
#     last_encounter_values = df.groupby(patient_id)[encounter_id].tail(1).values
#     return df[df[encounter_id].isin(last_encounter_values)]

def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
#     train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    common_denominator = 0.2
    sample_size = round(total_values * (common_denominator))
    train = df[df[patient_key].isin(unique_values[:sample_size*3])].reset_index(drop=True)
    validation = df[df[patient_key].isin(unique_values[sample_size*3:sample_size*4])].reset_index(drop=True)
    test = df[df[patient_key].isin(unique_values[sample_size*4:])].reset_index(drop=True)
    
    return train, validation, test

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    
    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        tf_categorical_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
            key=c, vocabulary_file = vocab_file_path, num_oov_buckets=1)
        
        # Keep the embedding dimensionality proportionate to the length of the vocab file with a max embedding size of 20
        dims = min(20, file_len(vocab_file_path))
        tf_categorical_feature_column = tf.feature_column.embedding_column(tf_categorical_feature_column, dimension=dims)
        
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std

def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=0, dtype=tf.float64, normalizer_fn=lambda x: normalize_numeric_with_zscore(x, MEAN, STD))
    
    return tf_numeric_feature

def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >=5 else 0).to_numpy()
    return student_binary_prediction
