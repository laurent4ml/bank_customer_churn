
"""
Customer churn test

This is the test and logging file for Customer Churn project
"""
import logging
import os
import pandas as pd
import joblib
import pytest
import numpy as np
from sklearn.model_selection import train_test_split
import customer_churn as cc

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/customer_churn_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name="random_df")
def fixture_random_df():
    '''
    random_df
    creating a dataframe based on randomly generated values

    Returns:
        dataframe: dataframe containing all data needed for the unit tests
    '''
    return pd.DataFrame({
                        'Gender': np.random.randint(low=0, high=2, size=1000),
                        'Marital_Status': np.random.randint(low=0, high=2, size=1000),
                        'Customer_Age': np.random.randint(low=20, high=100, size=1000),
                        'Dependent_count': np.random.randint(low=0, high=6, size=1000),
                        'Credit_Limit': np.random.randint(low=5000, high=100000,
                                                          size=1000),
                        'Total_Revolving_Bal': np.random.randint(low=20, high=100000,
                                                                 size=1000),
                        'Churn': np.random.randint(low=0, high=2, size=1000)})


@pytest.fixture(name="features_quant1")
def fixture_features_quant1():
    """
    features_quant1 creating a list of quant features

        Returns:
                list: list containing 2 relevant quant features
        """
    return ['Customer_Age', 'Dependent_count']


@pytest.fixture(name="features_quant2")
def fixture_features_quant2():
    """
    features_quant2 creating a list of quant features

        Returns:
                list: list containing 2 relevant quant features
        """
    return ['Credit_Limit', 'Total_Revolving_Bal']


@pytest.fixture(name="features_categorical")
def fixture_features_categorical():
    """
    features_categorical creating a list of categorical features

    Returns:
        list: list containing 2 relevant categorical features
    """
    return ['Gender', 'Marital_Status']


@pytest.fixture(name="features_keep")
def fixture_features_keep():
    """
    features_keep creating a list of features to use for model training

    Returns:
        list: list containing relevant categorical features
     """
    return ['Gender', 'Marital_Status', 'Customer_Age', 'Dependent_count',
            'Credit_Limit', 'Total_Revolving_Bal']


@pytest.fixture(name="features_all")
def fixture_features_all():
    """
    features_all creating a list of features to use for model training

    Returns:
        list: list containing all relevant categorical features
    """
    return ['Gender', 'Marital_Status', 'Customer_Age', 'Dependent_count',
            'Credit_Limit', 'Total_Revolving_Bal', 'Churn']

@pytest.fixture(name="test_size")
def fixture_test_size():
    '''
	test_size percentage of train_test_split to use for test

	Returns:
        decimal: percentage to use for test
    '''
    return 0.3

@pytest.fixture(name="result_path")
def fixture_result_path():
    '''
	result_path path to store results

    Returns:
        str: path to use to store results images
    '''
    return "images/results"

@pytest.fixture(name="sample_df")
def fixture_sample_df():
    '''
	sample_df dataframe used for testing

    Returns:
        dataframe: a sample dataframe for testing
    '''
    return pd.DataFrame({
        'Gender': ["Male", "Female", "Male", "Female"],
        'Marital_Status': ["Single", "Married", "Married", "Single"],
        'Customer_Age': [25, 35, 40, 45],
        'Dependent_count': [0, 1, 2, 1],
        'Credit_Limit': [5000, 25000, 40000, 20000],
        'Total_Revolving_Bal': [2000, 100, 4000, 3000],
        'Churn': [0, 1, 0, 1]
    })

@pytest.fixture(name="grid_params")
def fixture_grid_params():
    '''
    grid_params parameters used for GridSearchCV

    Returns:
        parameters for GridSearchCV
    '''
    return {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

@pytest.fixture(name="split_data")
def fixture_split_data(random_df, features_keep, features_categorical, test_size):
    '''
    split_data return train, test split datasets

	Returns:
		a set of 4 dataframes used for training and testing models
	'''
    output_df = cc.perform_feature_engineering(random_df, features_keep,
                                               features_categorical)
    return train_test_split(
        output_df, random_df['Churn'], test_size=test_size,
        random_state=42)


@pytest.fixture(name="model_rfc")
def fixture_model_rfc():
    '''
	model_rfc load rfc model from directory

    Returns:
        model: rfc model use for testing purposes
    '''
    return joblib.load('./test_models/rfc_model.pkl')

@pytest.fixture(name="model_lrc")
def fixture_model_lrc():
    '''
    model_lrc load lrc model from directory

    Returns:
        model: lrc model use for testing purposes
    '''
    return joblib.load('./test_models/logistic_model.pkl')

def test_clean_directory():
    '''
    test_clean_directory test cleaning directory which remove all files from
        a directory or create a new directory if it doesn't exists
    '''
    # clean directory of an existing path
    cc.clean_directory("images/eda/")

    # test that directory is empty
    try:
        assert not os.listdir("images/eda")
    except AssertionError as err:
        logging.error("Testing clean_directory: The directory is not empty")
        raise err

    # clean directory of a none existing path
    cc.clean_directory("some_dir")

    # test that new directory was created
    try:
        assert os.path.exists("some_dir") and os.path.isdir("some_dir")
    except AssertionError as err:
        logging.error("Testing clean_directory: The directory is not empty")
        raise err

def test_import():
    '''
    test_import test function for importing data
    '''
    dataframe = cc.import_data("./data/bank_data.csv")

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: the file doesn't have rows and columns")
        raise err

def test_eda(
        features_keep,
        features_quant1,
        features_categorical,
        features_quant2):
    '''
    test_eda test perform eda function
    '''
    # check that files with the correct names exist in the image directory
    # first let's setup a df with only 3 columns
    test_path = "images/test_eda/"
    full_df = cc.import_data("./data/bank_data.csv")
    small_df = full_df[features_keep]
    # second create a new directory to store testing images
    cc.clean_directory(test_path)

    cc.perform_eda(small_df, features_quant1,
                   features_categorical, features_quant2, test_path)

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error("Testing test_eda: the test directory is not created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)
    num_cols = len(features_quant1) + len(features_categorical)
    num_cols += len(features_quant2) + 1
    try:
        assert num_files == num_cols
    except AssertionError as err:
        logging.error(
            "Testing test_eda: incorrect number of files in directory")
        raise err

def test_encoder_helper(sample_df, features_categorical):
    '''
    test encoder helper
    '''
    dataframe = cc.encoder_helper(sample_df, features_categorical)

    try:
        assert 'Gender_Churn' in dataframe.columns
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: column Gender_Churn does not exist")
        raise err

    try:
        assert 'Marital_Status_Churn' in dataframe.columns
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: column Gender_Churn does not exist")
        raise err

    try:
        assert dataframe['Marital_Status_Churn'][1] == 0.5
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: Marital_Status_Churn error")
        raise err

    try:
        assert dataframe['Gender_Churn'][1] == 1
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: Gender_Churn error for loc 1")
        raise err

    try:
        assert dataframe['Gender_Churn'][0] == 0
    except AssertionError as err:
        logging.error(
            "Testing test_encoder_helper: Gender_Churn error for loc 0")
        raise err

def test_perform_feature_engineering(
        sample_df,
        features_keep,
        features_categorical):
    '''
    test perform_feature_engineering
    '''
    # check that the resulting dataframe has the correct number of columns
    dataframe = cc.perform_feature_engineering(sample_df, features_keep,
                                               features_categorical)

    expected = ['Gender', 'Marital_Status', 'Customer_Age',
                'Dependent_count', 'Credit_Limit', 'Total_Revolving_Bal']

    try:
        assert all([a == b for a, b in zip(dataframe.columns, expected)])
    except AssertionError as err:
        logging.error("Testing test_encoder_helper: error in keep cols")
        raise err

def test_train_model_rfc(split_data, grid_params):
    '''
    test train_model_rfc
    '''
    test_path = "test_model_rfc/"
    cc.clean_directory(test_path)

    x_train, _, y_train, _ = split_data

    cc.train_model_rfc(x_train, y_train, grid_params, pth=test_path)

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error(
            "Testing test_train_model_rfc: test_path directory isn't created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 1
    except AssertionError as err:
        logging.error(
            "Testing test_train_model_rfc: missing files in directory")
        raise err

    try:
        assert all_files == ['rfc_model.pkl']
    except AssertionError as err:
        logging.error("Testing test_train_model_rfc: incorrect file")
        raise err

def test_train_model_lrc(split_data):
    '''
    test train_model_lrc
    '''
    test_path = "test_model_lrc/"
    cc.clean_directory(test_path)

    x_train, _, y_train, _ = split_data

    cc.train_model_lrc(x_train, y_train, pth=test_path)

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error(
            "Testing test_train_model_lrc: test_path directory isn't created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 1
    except AssertionError as err:
        logging.error(
            "Testing test_train_model_lrc: missing files in directory")
        raise err

    try:
        assert all_files == ['logistic_model.pkl']
    except AssertionError as err:
        logging.error("Testing test_train_model_lrc: incorrect file")
        raise err

def test_classification_report_image(split_data, model_rfc, model_lrc):
    '''
    test classification_report_image
    '''
    test_path = "images/test_report/"
    cc.clean_directory(test_path)

    x_train, x_test, y_train, y_test = split_data

    y_train_preds_rf = model_rfc.predict(x_train)
    y_test_preds_rf = model_rfc.predict(x_test)
    y_train_preds_lr = model_lrc.predict(x_train)
    y_test_preds_lr = model_lrc.predict(x_test)

    # Create Random Forest Train and Test plot
    cc.classification_report_image(
        'Random Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        test_path)

    # Create Classification Report plot
    cc.classification_report_image(
        'Logistic Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        test_path)

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error(
            "Testing test_report_image: test_path directory isn't created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 2
    except AssertionError as err:
        logging.error("Testing test_report_image: missing files in directory")
        raise err

def test_feature_importance_plot(split_data, model_rfc):
    '''
    test feature importance plot
    '''
    test_path = "images/test_fip"
    cc.clean_directory(test_path)
    x_train, _, _, _ = split_data
    cc.feature_importance_plot(model_rfc, x_train,
                               test_path + '/feature_importance.png')

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error("Testing test_roc: test_path directory isn't created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 1
    except AssertionError as err:
        logging.error("Testing test_roc: missing files in directory")
        raise err

    try:
        assert all_files == ['feature_importance.png']
    except AssertionError as err:
        logging.error("Testing test_feature_importance_plot: incorrect file")
        raise err

def test_save_shap_report(split_data, model_rfc):
    '''
    test save SHAP report
    '''
    test_path = "images/test_shap"
    cc.clean_directory(test_path)
    _, x_test, _, _ = split_data

    cc.save_shap_report(model_rfc, x_test, test_path + '/shap.png')

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error("Testing test_roc: test_path directory isn't created")
        raise err

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 1
    except AssertionError as err:
        logging.error("Testing test_save_shap_report: missing files in directory")
        raise err

    try:
        assert all_files == ['shap.png']
    except AssertionError as err:
        logging.error("Testing test_save_shap_report: incorrect file in directory")
        raise err

def test_receiver_operating_characteristic(split_data, model_rfc, model_lrc):
    '''
    test receiver operating characteristic function
    '''
    test_path = "images/test_roc/"
    cc.clean_directory(test_path)

    try:
        assert os.path.exists(test_path) and os.path.isdir(test_path)
    except AssertionError as err:
        logging.error("Testing test_roc: test_path directory isn't created")
        raise err

    _, x_test, _, y_test = split_data

    cc.receiver_operating_characteristic(model_lrc, model_rfc, x_test, y_test,
                                         pth=test_path)

    all_files = os.listdir(test_path)
    num_files = len(all_files)

    try:
        assert num_files == 1
    except AssertionError as err:
        logging.error("Testing test_roc: missing files in directory")
        raise err

    try:
        assert all_files == ['roc.png']
    except AssertionError as err:
        logging.error("Testing test_roc: incorrect file in directory")
        raise err

