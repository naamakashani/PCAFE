# -*- coding: utf-8 -*-

import numpy as np
import gzip
import struct
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import csv
import pandas as pd
import torch
from sklearn.utils import resample
from ucimlrepo import fetch_ucirepo
import scipy.io


def add_noise(X, noise_std=0.01):
    """
    Add Gaussian noise to the input features.

    Parameters:
    - X: Input features (numpy array).
    - noise_std: Standard deviation of the Gaussian noise.

    Returns:
    - X_noisy: Input features with added noise.
    """
    noise = np.random.normal(loc=0, scale=noise_std, size=X.shape)
    X_noisy = X + noise
    return X_noisy


def balance_class(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, add_noise(X[noisy_indices], noise_std)], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced



def load_ucimlrepo():
    # fetch dataset
    diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)
    X = diabetic_retinopathy_debrecen.data.features.to_numpy()
    y = diabetic_retinopathy_debrecen.data.targets.to_numpy()
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)
    return X, y, diabetic_retinopathy_debrecen.metadata.num_features, diabetic_retinopathy_debrecen.metadata.num_features


def load_fetal():
    file_path = r'/data/fetal_health_None.csv'
    df = pd.read_csv(file_path)
    Y = df['fetal_health'].to_numpy().reshape(-1)
    X = df.drop(df.columns[-1], axis=1).to_numpy()
    return X, Y, Y, len(X[0])




def balance_class(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]

    # Get indices of samples belonging to each class
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]

    # Calculate the difference in sample counts
    minority_count = len(minority_indices)
    majority_count = len(majority_indices)
    count_diff = majority_count - minority_count

    # Add noise to the features of the minority class to balance the dataset
    if count_diff > 0:
        # Randomly sample indices from the minority class to add noise
        noisy_indices = np.random.choice(minority_indices, count_diff, replace=True)
        # Add noise to the features of the selected samples
        X_balanced = np.concatenate([X, X[noisy_indices]], axis=0)
        y_balanced = np.concatenate([y, y[noisy_indices]], axis=0)
    else:
        X_balanced = X.copy()  # No need for balancing, as classes are already balanced
        y_balanced = y.copy()
    return X_balanced, y_balanced


def load_text_data():
    sentences_of_diabetes_diagnosis = [
        'Monitoring blood glucose levels is crucial for diabetes management.',
        'Type 1 diabetes is an autoimmune condition that typically develops in childhood.',
        'Type 2 diabetes can often be managed with lifestyle changes and medication.',
        'Continuous glucose monitoring devices help individuals track their blood sugar levels.',
        'Regular exercise can improve insulin sensitivity in individuals with diabetes.',
        'Diabetes education programs teach patients about managing their condition effectively.',
        'A balanced diet rich in whole grains and vegetables is recommended for diabetes prevention.',
        'Diabetes mellitus is characterized by high levels of sugar in the blood.',
        'Elevated fasting blood glucose levels indicate possible diabetes.',
        'Insulin resistance is a common precursor to type 2 diabetes.',
        'Family history of diabetes increases the risk of developing the condition.',
        'Excessive thirst and frequent urination are classic symptoms of diabetes.',
        'Hemoglobin A1c levels above 6.5% suggest diabetes mellitus.',
        'Obesity and sedentary lifestyle contribute to the onset of diabetes.',
        'Diabetic retinopathy is a complication that affects the eyes in diabetes patients.',
        'Neuropathy, characterized by tingling or numbness, is a common diabetes-related complication.',
        'Gestational diabetes can occur during pregnancy and requires monitoring.',
        'Ketoacidosis is a serious condition that can occur in untreated type 1 diabetes.', 'high suger',
        'problem in suger', "I have high blood sugar.",
        "I need insulin shots.",
        "My doctor says I have diabetes.",
        "I feel thirsty all the time.",
        "I have to check my blood sugar every day.",
        "My blood sugar is often too high.",
        "I need to take diabetes medication.",
        "I have to watch what I eat because of diabetes.",
        "My blood sugar levels are unstable.",
        "I visit my doctor regularly for my diabetes.",
        "I experience frequent urination.",
        "I have numbness in my feet.",
        "I was diagnosed with diabetes last year.",
        "I follow a special diet for diabetes.",
        "My doctor monitors my blood sugar levels closely.",
        "I have type 1 diabetes.",
        "I have type 2 diabetes.",
        "I use a continuous glucose monitor.",
        "My hemoglobin A1c levels are high.",
        "I am managing my diabetes with medication.",
        "I need to monitor my blood sugar levels every day.",
        "I follow a diabetic diet to manage my condition.",
        "My blood sugar spikes after meals.",
        "I have regular check-ups for my diabetes.",
        "I have to be careful with my carbohydrate intake.",
        "I experience fatigue due to my diabetes.",
        "I need to take medication to control my diabetes.",
        "I have diabetes-related eye issues.",
        "I have been living with diabetes for several years.",
        "I carry glucose tablets for emergencies.",
        "My blood sugar drops if I don't eat regularly.",
        "I have a family history of diabetes.",
        "I am on a low-sugar diet.",
        "I use an insulin pump to manage my diabetes.",
        "I get my HbA1c levels checked regularly.",
        "I have to avoid sugary foods because of my diabetes.",
        "I need to watch my blood sugar closely.",
        "I have diabetes-related nerve pain.",
        "I take insulin injections daily.",
        "I have a special meal plan for my diabetes",
        "I need to be careful with my diet because of diabetes.",
        "I have to plan my meals carefully.",
        "I experience dizziness when my blood sugar is low.",
        "I keep a record of my blood sugar readings.",
        "I attend diabetes education classes.",
        "I have to take care of my feet because of diabetes.",
        "I use special shoes for my diabetic feet.",
        "I check my blood sugar before and after exercising.",
        "I need to avoid high-sugar foods.",
        "I visit my endocrinologist regularly.",
        "I need to eat snacks to avoid low blood sugar.",
        "I am careful about my carbohydrate intake.",
        "I follow my doctor's advice to manage my diabetes.",
        "I have a diabetes management plan.",
        "I need to balance my insulin doses with my food intake.",
        "I check my blood sugar levels multiple times a day.",
        "I don't need to worry about my blood sugar levels.",
             "I can eat anything without worrying about diabetes.",
        "I have a healthy and active lifestyle.",
        "I don't have to take medication for my blood sugar.",
        "I have never experienced symptoms of diabetes.",
        "I feel great and have no health issues.",
        "I don't need to follow a special diet.",
        "I enjoy a variety of foods without restrictions.",
        "I don't need to check my blood sugar levels.",
        "I have normal health check-ups.",
        "I don't have any family members with diabetes.",
        "I have a balanced diet and stay active.",
        "I don't need to visit the doctor for blood sugar issues.",
        "I maintain my weight and stay fit.",
        "I can eat sweets in moderation without any issues.",
        "I don't need to monitor my health constantly.",
        "I have good overall health and well-being.",
        "I enjoy good health and physical fitness.",
        "I don't have any chronic health conditions."

    ]
    sentences_of_health_people = [
        'Regular physical activity is important for maintaining overall health.',
        'A balanced diet rich in fruits and vegetables supports good health.',
        'Adequate sleep is essential for physical and mental well-being.',
        'Maintaining a healthy weight reduces the risk of chronic diseases.',
        'Hydrating properly throughout the day supports optimal bodily functions.',
        'Regular medical check-ups help detect early signs of health issues.',
        'Practicing stress management techniques promotes mental resilience.',
        'Avoiding smoking and excessive alcohol consumption improves health outcomes.',
        'Social connections and community engagement contribute to overall well-being.',
        'Healthy habits include brushing teeth twice daily and flossing regularly.',
        'Normal fasting blood sugar levels typically range between 70 to 100 milligrams per deciliter (mg/dL).',
        'Postprandial (after-meal) blood sugar levels in healthy individuals usually stay below 140 mg/dL.',
        'HbA1c levels below 5.7% are considered normal, indicating good long-term blood sugar control.',
        'Non-diabetic individuals maintain stable blood sugar levels throughout the day.',
        'Physical activity can help regulate blood sugar levels even in individuals without diabetes.',
        'Healthy eating habits can prevent blood sugar spikes and maintain stable glucose levels.',
        'Regular monitoring of blood sugar levels can help individuals understand their body\'s glucose regulation.',
        'Stress management and adequate sleep contribute to maintaining healthy blood sugar levels.',
        'Age and genetics can influence individual variations in normal blood sugar levels.',
        'Maintaining a healthy lifestyle reduces the risk of developing abnormal blood sugar levels.',
        "I have normal blood sugar levels.",
        "I do not have diabetes.",
        "My blood sugar is always stable.",
        "I do not need to take insulin.",
        "I eat a regular diet without restrictions.",
        "I do not need to monitor my blood sugar.",
        "I have never been diagnosed with diabetes.",
        "I feel healthy and energetic.",
        "I do not experience excessive thirst.",
        "I have no issues with frequent urination.",
        "I maintain a healthy weight.",
        "I exercise regularly and eat well.",
        "I do not have any family history of diabetes.",
        "I do not take any diabetes medication.",
        "My doctor says I am healthy.",
        "I have no symptoms of diabetes.",
        "My blood tests are always normal.",
        "I have good overall health.",
        "I do not have any health problems.",
        "I maintain a balanced diet and active lifestyle.",
        "I don't have to monitor my blood sugar.",
             "I can eat a variety of foods without restrictions.",
        "I have no problems with my blood sugar levels.",
        "I feel energetic and healthy.",
        "I don't take any medication for blood sugar.",
        "I have never had issues with high blood sugar.",
        "My health check-ups are always normal.",
        "I don't need to follow a special diet.",
        "I don't have any symptoms of diabetes.",
        "I maintain a balanced diet without worrying about sugar.",
        "I have no need for insulin or other diabetes medication.",
        "I do not experience any diabetic symptoms.",
        "I enjoy regular physical activity.",
        "I have never been told I have high blood sugar.",
        "I have no need to track my blood sugar levels.",
        "I can enjoy sweets in moderation without problems.",
        "My blood sugar levels are always within normal range.",
        "I don't have a family history of diabetes.",
        "I don't worry about my blood sugar levels.",
        "I maintain my health through regular exercise and a balanced diet."

    ]
    # open diabetes_prediction_text
    diabetes_prediction_text = pd.read_csv(
        r'/DATA/diabetes_prediction_text.csv')

    # take 100 samples from diabetes_prediction_text
    diabetes_prediction_text = diabetes_prediction_text.sample(n=2000, random_state=1)
    diabetes_prediction_text['text'] = np.where(diabetes_prediction_text['diabetes'] == 0,
                                                np.random.choice(sentences_of_health_people,
                                                                 diabetes_prediction_text.shape[0]),
                                                np.random.choice(sentences_of_diabetes_diagnosis,
                                                                 diabetes_prediction_text.shape[0]))

    labels = diabetes_prediction_text.iloc[:, -1].values.astype(int)  # Labels as numpy array of integers
    diabetes_prediction_text = diabetes_prediction_text.drop(columns=['diabetes'])  # Drop the label column
    # Identify numeric and text columns
    numeric_columns = diabetes_prediction_text.select_dtypes(include=np.number).columns
    text_columns = diabetes_prediction_text.columns.difference(numeric_columns)

    numeric_features = diabetes_prediction_text.loc[:, numeric_columns].values
    text_features = diabetes_prediction_text.loc[:, text_columns].astype(str).values

    # Assuming the last column is the label
    y = labels
    # Get numeric and text column indices
    numeric_column_indices = [diabetes_prediction_text.columns.get_loc(col) for col in numeric_columns]
    text_column_indices = [diabetes_prediction_text.columns.get_loc(col) for col in text_columns]
    #name of the columns
    column_names = diabetes_prediction_text.columns

    return numeric_features, text_features, y, numeric_features.shape[1] + text_features.shape[
        1], text_column_indices, numeric_column_indices, column_names





def import_breast():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    breast_cancer_wisconsin_prognostic = fetch_ucirepo(id=16)

    # data (as pandas dataframes)
    X = breast_cancer_wisconsin_prognostic.data.features.to_numpy()
    y = breast_cancer_wisconsin_prognostic.data.targets.to_numpy()
    y[y == 'R'] = 1
    y[y == 'N'] = 0
    X[X == 'nan'] = 0
    X = np.nan_to_num(X, nan=0)
    y = y.squeeze()
    y = y.tolist()
    y = np.array(y)

    return X, y, breast_cancer_wisconsin_prognostic.metadata.num_features, breast_cancer_wisconsin_prognostic.metadata.num_features


def create_data():
    # Number of points to generate
    num_points = 100
    # Generate random x values
    x1_values = np.random.uniform(low=-30, high=30, size=num_points)
    x2_values = np.random.uniform(low=-30, high=30, size=num_points)
    x3_values = np.random.uniform(low=-30, high=30, size=num_points)
    # Create y values based on the decision boundary if x+y > 9 then 1 else 0
    labels = np.where((x1_values + x2_values + x3_values > 10), 1, 0)
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values, x3_values))
    return x, labels, x1_values, 3


def create():
    # Set a random seed for reproducibility

    # Number of points to generate
    num_points = 100

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)

    # Create a scatter plot of the dataset with color-coded labels
    plt.scatter(x1_values, x2_values, c=labels, cmap='viridis', marker='o', label='Data Points')
    # Split the data into training and testing sets
    x = np.column_stack((x1_values, x2_values))
    return x, labels, x1_values, 3


def balance_class_multi(X, y, noise_std=0.01):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_class_count = np.max(class_counts)

    # Calculate the difference in sample counts for each class
    count_diff = max_class_count - class_counts

    # Initialize arrays to store balanced data
    X_balanced = X.copy()
    y_balanced = y.copy()

    # Add noise to the features of the minority classes to balance the dataset
    for minority_class, diff in zip(unique_classes, count_diff):
        if diff > 0:
            # Get indices of samples belonging to the current minority class
            minority_indices = np.where(y == minority_class)[0]

            # Randomly sample indices from the minority class to add noise
            noisy_indices = np.random.choice(minority_indices, diff, replace=True)

            # Add noise to the features of the selected samples
            X_balanced = np.concatenate([X_balanced, add_noise(X[noisy_indices], noise_std)], axis=0)
            y_balanced = np.concatenate([y_balanced, y[noisy_indices]], axis=0)

    return X_balanced, y_balanced


def create_n_dim():
    # Number of points to generate
    num_points = 2000

    # Generate random x values
    x1_values = np.random.uniform(low=0, high=30, size=num_points)

    # Create y values based on the decision boundary y=-x with some random noise
    x2_values = -x1_values + np.random.normal(0, 2, size=num_points)

    # Create labels based on the side of the decision boundary
    labels = np.where(x2_values > -1 * x1_values, 1, 0)
    # create numpy of zeros
    X = np.zeros((num_points, 10))
    i = 0
    while i < num_points:
        # choose random index to assign x1 and x2 values
        index = np.random.randint(0, 10)
        # assign x1 to index for 5 samples
        X[i][index] = x1_values[i]
        X[i + 1][index] = x1_values[i + 1]
        X[i + 2][index] = x1_values[i + 2]
        X[i + 3][index] = x1_values[i + 3]
        X[i + 4][index] = x1_values[i + 4]
        # choose random index to assign x2 that is not the same as x1
        index2 = np.random.randint(0, 10)
        while index2 == index:
            index2 = np.random.randint(0, 10)
        X[i][index2] = x2_values[i]
        X[i + 1][index2] = x2_values[i + 1]
        X[i + 2][index2] = x2_values[i + 2]
        X[i + 3][index2] = x2_values[i + 3]
        X[i + 4][index2] = x2_values[i + 4]
        i += 5
    question_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    return X, labels, question_names, 10


def load_gisetta():
    data_path = "/RL/extra/gisette/gisette_train.data"
    labels_path = "/RL/extra/gisette/gisette_train.labels"
    data = []
    labels = []
    with open(labels_path, newline='') as file:
        # read line by line
        for line in file:
            if int(line) == -1:
                labels.append(0)
            else:
                labels.append(1)
    with open(data_path, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            sample = []
            columns = row[0].split(' ')
            for i in range(len(columns) - 2):
                sample.append(float(columns[i]))
            data.append(sample)
    X = np.array(data)
    y = np.array(labels)
    return X, y, y, len(sample)




def load_diabetes():
    data = []
    labels = []
    file_path_clean = '/RL/extra/diabetes/diabetes_clean.csv'
    # Open the CSV file
    with open(file_path_clean, newline='') as csvfile:
        # Create a CSV reader
        csv_reader = csv.reader(csvfile)
        for line in csv_reader:
            # if it the first line (header) skip itda
            if line[0] == 'gender':
                # save the header in npy array for later use
                question_names = np.array(line)
                continue
            # columns = [column.split(',') for column in line]
            columns = line
            columns_without_label = columns[0:-1]
            if columns_without_label[0] == "Female":
                columns_without_label[0] = 0
            else:
                columns_without_label[0] = 1
            if columns_without_label[4] == "never":
                columns_without_label[4] = 0
            if columns_without_label[4] == "former":
                columns_without_label[4] = 1
            if columns_without_label[4] == "current":
                columns_without_label[4] = 2
            if columns_without_label[4] == "No Info":
                columns_without_label[4] = 3
            if columns_without_label[4] == "not current":
                columns_without_label[4] = 4
            if columns_without_label[4] == "ever":
                columns_without_label[4] = 5
            for i in range(len(columns_without_label)):
                if columns_without_label[i] != '':
                    columns_without_label[i] = float(columns_without_label[i])
                else:
                    columns_without_label[i] = 0

            data.append(columns_without_label)

            labels.append(int(float(columns[-1])))

    # Convert zero_list to a NumPy array
    X = np.array(data)
    y = np.array(labels)
    # X,y=balance_class(X, y)
    # y = y.reshape((-1, 1))
    #
    # data_labels = np.concatenate((X, y), axis=1)
    # data_labels_df = pd.DataFrame(data_labels)
    #
    # data_labels_df.to_csv(r'C:\Users\kashann\PycharmProjects\choiceMira\RL\extra\diabetes\diabetes_prediction_text.csv', index=False)
    return X, y, question_names, len(columns_without_label)


def create_demo():
    np.random.seed(34)
    Xs1 = np.random.normal(loc=1, scale=0.5, size=(300, 5))
    Ys1 = -2 * Xs1[:, 0] + 1 * Xs1[:, 1] - 0.5 * Xs1[:, 2]
    Xs2 = np.random.normal(loc=-1, scale=0.5, size=(300, 5))
    Ys2 = -0.5 * Xs2[:, 2] + 1 * Xs2[:, 3] - 2 * Xs2[:, 4]
    X_data = np.concatenate((Xs1, Xs2), axis=0)
    Y_data = np.concatenate((Ys1.reshape(-1, 1), Ys2.reshape(-1, 1)), axis=0)
    Y_data = Y_data - Y_data.min()
    Y_data = Y_data / Y_data.max()
    case_labels = np.concatenate((np.array([1] * 300), np.array([2] * 300)))
    Y_data = np.concatenate((Y_data, case_labels.reshape(-1, 1)), axis=1)
    Y_data = Y_data[:, 0].reshape(-1, 1)
    return X_data, Y_data, 5, 5


