'''
A data science solution process including:

EDA
Feature Engineering (including encoding of categorical variables)
Model Training
Prediction
Model Evaluation
'''
# from sklearn.preprocessing import normalize
import logging
import os
from pyinstrument import Profiler
from omegaconf import DictConfig
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split
# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/customer_churn.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config: DictConfig):
    '''
    run a full end to end machine learning process

    Input:
        config: (yaml) configs are loaded from a yaml file

    Output:
        None
    '''
    profiler = Profiler()
    profiler.start()
    logging.info("preparing images and models directories")
    dirs = ['images/eda/', 'images/results/', 'images/reports/', 'models/']
    for directory in dirs:
        clean_directory(directory)

    # importing the dataset into a pandas dataframe
    logging.info("importing data")
    try:
        input_df = import_data("./data/bank_data.csv")
    except AssertionError as err:
        logging.error("error importing file: " + err)
        raise err

    # Churn column definition
    # churn column is set to 0 if column Attrition_Flag is set to Existing Customer
    # otherwise its set to 1
    input_df['Churn'] = input_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # perform Explanatory Data Analysis (EDA) and plot visualizations
    logging.info("perform eda")
    perform_eda(input_df, config.features.quant1, config.features.categorical,
                config.features.quant2, "images/eda")

    # perform feature engineering
    logging.info("perform feature engineering")
    try:
        output_df = perform_feature_engineering(input_df,
                                                keep_cols=config.features.keep,
                                                cat_cols = config.features.categorical)
    except AssertionError as err:
        logging.error("perform feature engineering error: " + err)
        raise err

    x_train, x_test, y_train, y_test = train_test_split(
        output_df, input_df['Churn'], test_size=config.test_size, random_state=42)

    # train models
    logging.info("training models")
    print("training models")
    train_model_lrc(x_train, y_train, pth = "models")
    grid_params = {
        'n_estimators': config.grid_params.n_estimators,
        'max_features': config.grid_params.max_features,
        'max_depth': config.grid_params.max_depth,
        'criterion': config.grid_params.criterion
    }
    train_model_rfc(x_train, y_train, grid_params, pth = "models")

	# loading models
    print("loading models")
    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    print("receiver operating characteristic")
    receiver_operating_characteristic(lr_model, rfc_model, x_test, y_test,
                                    pth=config.result_path)

    y_train_preds_rf = rfc_model.predict(x_train)
    y_test_preds_rf = rfc_model.predict(x_test)
    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)

    # Create Random Forest Train and Test plot
    print("generating classification report for Random Forest model")
    classification_report_image(
        'Random Forest',
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'images/reports'
    )

    # Create Classification Report plot
    print("generating classification report for Logistic Regression model")
    classification_report_image(
        'Logistic Regression',
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'images/reports'
    )

    # Create SHAP summary plot for random forest classifier
    print("save shap report")
    save_shap_report(rfc_model, x_test, config.result_path + '/shap.png')

    # Create Feature Importance plot for random forest classifier
    print("feature importance plot")
    feature_importance_plot(rfc_model, x_train,
                    config.result_path + '/feature_importance.png')

    logging.info("full cycle completed")
    print("full cycle completed")
    profiler.stop()
    profiler.print()

def clean_directory(path):
    '''
    clean directory creates a folder if it doesn't exists
    if folder exists it removes all files that it contains

    input:
        path: (str) path to clean

    output:
        None
    '''
    logging.info("cleaning directory %s", path)
    if not os.path.exists(path):
        logging.info("created directory %s", path)
        os.mkdir(path, 755)
    else:
        for file_name in os.listdir(path):
            # construct full file path
            file = path + file_name
            if os.path.isfile(file):
                logging.info("deleting file: %s", file)
                os.remove(file)
            else:
                logging.info("not a file: %s", file)

def import_data(pth):
    '''
    returns dataframe from the csv file found at pth

    input:
        pth: (str) a path to the csv file

    output:
        dataframe: (dataframe) data return from csv file
    '''
    try:
        assert len(pth) > 0
    except AssertionError as err:
        logging.error("ERROR: importing data, file not defined")
        raise err

    try:
        dataframe = pd.read_csv(pth)
        logging.info("SUCCESS: importing data")
    except FileNotFoundError as err:
        logging.error("ERROR: importing data, the file wasn't found")
        raise err

    return dataframe

def perform_eda(dataframe, quant1_columns, cat_columns, quant2_columns,
                eda_path = "images/eda"):
    '''
    perform exploratory data analysis and plot visualization
    save figures to images/eda folder

    input:
        dataframe: (dataframe) data we want to explore
        quant1_columns: (list) quantity columns first sample
        cat_columns: (list) categorical columns
        quant2_columns: (list) quantity columns second sample
        eda_path: (str) path to save images

    output:
        None
    '''
    logging.info("Explanatory Data Analysis - Start")
    logging.info(dataframe.head())
    logging.info(dataframe.shape)
    logging.info(dataframe.isnull().sum())
    logging.info(dataframe.describe())

    for col in quant1_columns:
        title = col.replace("_", " ") + " Histogram"
        plt.close('all')
        plt.figure(figsize=(20, 10))
        plt.title(title)
        dataframe[col].hist()
        plt.savefig(eda_path + '/' + col.lower() + '.png')

    for col in cat_columns:
        title = col.replace("_", " ")
        plt.close('all')
        plt.figure(figsize=(20, 10))
        plt.title(title)
        dataframe[col].value_counts('normalize').plot(kind='bar')
        plt.savefig(eda_path + '/' + col.lower() +'.png')

    # distplot is deprecated. Use histplot instead
    # sns.distplot(df['Total_Trans_Ct']);
    # Show distributions of 'Total_Trans_Ct' and add a smooth curve obtained using a
    # kernel density estimate
    for col in quant2_columns:
        title = col.replace("_", " ") + " Histplot"
        plt.close('all')
        plt.figure(figsize=(20, 10))
        plt.title(title)
        sns.histplot(dataframe[col], stat='density', kde=True)
        plt.savefig(eda_path + '/' + col.lower() + '.png')

    # Create a Columns Correlation Heatmap
    plt.close('all')
    plt.figure(figsize=(20, 20))
    # Create plot title
    plt.title("Columns Correlations")
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(eda_path + '/columns_correlation.png')

def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category

    input:
        dataframe: (dataframe) contains categorical columns
        category_lst: (list) columns that contain categorical features

    output:
        dataframe: pandas dataframe with new columns "_Churn"
    '''
    try:
        assert len(category_lst) > 0
    except AssertionError as err:
        logging.error("encoder_helper: category_lst is empty")
        raise err

    for category in category_lst:
        logging.info("encoding %s", category)
        lst = []
        groups = dataframe.groupby(category).mean()['Churn']
        for val in dataframe[category]:
            lst.append(groups.loc[val])
        category_churn = category + "_Churn"
        dataframe[category_churn] = lst

    return dataframe

def perform_feature_engineering(dataframe, keep_cols, cat_cols):
    '''
    Feature Engineering Steps including encoding some columns

    input:
        dataframe: (dataframe) data we want to perform feature engineering on
        keep_cols: (list) all features needed to train models
        cat_cols: (list) categorical columns

    output:
        x_df: dataframe with encoded columns
    '''
    try:
        dataframe = encoder_helper(dataframe, cat_cols)
    except AssertionError as err:
        logging.error("encoder_helper error: " + err)
        raise err
    # removing unnecessary features
    dataframe = dataframe[keep_cols]
    logging.info(dataframe.head())
    return dataframe

def classification_report_image(
        title,
        y_train,
        y_test,
        y_train_preds,
        y_test_preds,
        pth = "images/reports"):
    '''
    produces and saves a classification report into an image folder as image

    input:
        title: (str) classification model name
        y_train: (dataframe) training response values
        y_test: (dataframe) test response values
        y_train_preds: (dataframe) training predictions from model
        y_test_preds: (dataframe) test predictions from model
        pth: (str) path to store images into

    output:
        None
    '''
    title1 = title + " Train"
    title2 = title + " Test"
    image = title.replace(" ", "_").lower() + '.png'
    text_kwargs = dict(va='center', fontsize=18, color='C1', fontproperties='monospace')
    plt.close('all')
    plt.rc('figure', figsize=(6, 7))
    plt.text(0.01, 0.1, str(title1), **text_kwargs)
    plt.text(0.01, 0.2, str(classification_report(y_train, y_train_preds)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(title2), **text_kwargs)
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(pth + '/' + image)

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in output path

    input:
        model: model object containing feature_importances_
        x_data: (dataframe) pandas dataframe of X values
        output_pth: (str) path to store the figure

    output:
        None
    '''
    logging.info("feature importance plot start")
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    plt.close('all')
    plt.figure(figsize=(20, 20))
    # Create plot title
    plt.title("Feature Importance", fontsize=20)
    plt.ylabel('Importance', fontsize=20)
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)

def train_model_rfc(x_train, y_train, grid_params, pth = "models"):
    '''
    train, store model results

    input:
        x_train: (dataframe) training vector of shape (n_train_samples, n_features)
        y_train: (dataframe) vector relative to x_train of shape (n_train_samples,)
        grid_params: (array) GridSearchCV parameters
        pth: (string) path to store models in

    output:
        None
    '''
    rfc = RandomForestClassifier(random_state=42)

	# GridSearchCV is a technique for finding the optimal parameter values from a given
	# set of parameters in a grid.
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=grid_params, cv=5)
    cv_rfc.fit(x_train, y_train)

    # save best random forest model
    joblib.dump(cv_rfc.best_estimator_, pth + '/rfc_model.pkl')

def train_model_lrc(x_train, y_train, pth = "models"):
    '''
    train, store model results

    input:
        x_train: (dataframe) training vector of shape (n_train_samples, n_features)
        y_train: (dataframe) vector relative to x_train of shape (n_train_samples,)
        pth: (str) path to store models in

    output:
        None
    '''
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(x_train, y_train)
    # save the logistic regression model
    joblib.dump(lrc, pth + '/logistic_model.pkl')

def receiver_operating_characteristic(model1, model2, x_test, y_test, pth):
    '''
    Create Receiver Operating Characteristic plot

    The more that a ROC curve hugs the top left corner of the plot, the better the model
    does at classifying the data into categories.

    Input:
        model1: first model to plot on the ROC graph
        model2: second model to plot on the ROC graph
        x_test: (dataframe) contains X test data
        y_test: (dataframe) contains y test data
        pth: (str) path to store results

    Output:
        None
    '''
    lrc_disp = plot_roc_curve(model1, x_test, y_test)

    plt.close('all')
    plt.figure(figsize=(15, 8))
    # Create plot title
    plt.title("Receiver Operating Characteristic", fontsize=20)
    # Axes object to plot on
    axe = plt.gca()
    plot_roc_curve(model2, x_test, y_test, ax=axe, alpha=0.8)
    lrc_disp.plot(ax=axe, alpha=0.8)
    try:
        plt.savefig(pth + '/roc.png')
    except PermissionError as err:
        logging.error("roc: test_path permission error")
        raise err

def save_shap_report(model, x_data, pth):
    '''
    save SHAP report to a image directory

    Input:
        model: (model) used to explain most relevant features
        x_data: (dataframe) contains test data
        pth: (string) path to store SHAP report image

    Output:
        None
    '''
    plt.close('all')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(pth, dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    main()
