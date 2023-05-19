import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def separate_classes(data):
    # create a list of class names
    classes = []
    for i in range(len(data)):
        if data[i][-1] not in classes:
            classes.append(data[i][-1])
    for i in range(len(classes)):
        for j in range(len(data)):
            if data[j][-1] == classes[i]:
                data[j][-1] = i

    # create a dictionary to separate classes
    separated = {}
    for i in range(len(data)):
        vector = data[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(np.array(vector))
    for key, value in separated.items():
        separated[key] = (np.array(value))

    return separated, classes


def y_prob(data_class, data_full, class_names):
    # create a dictionary of the probability of class occurrence
    class_name_list = [name for name in class_names]
    class_mean_list = [len(data_class[value]) / len(data_full) for value in data_class]
    prob_dict = {class_name_list[i]: class_mean_list[i] for i in range(len(class_name_list))}
    return prob_dict


def mean_calculate(X, classes):
    # create a dictionary for the mean values of each class
    mean_dictionary = {}
    for i in X:
        # n_features needs to have -1 since the matrix includes y values
        m_samples, n_features = X[i].shape
        mean_list = []
        for j in range(n_features - 1):
            mean_list.append((1 / m_samples) * np.sum(X[i][:, j]))
        mean_dictionary[classes[i]] = mean_list
    return mean_dictionary


def standard_deviation_calculate(X, mean, classes):
    # create a dictionary for the standard deviation values of each class
    standard_deviation_dictionary = {}
    for i in X:
        # n_features needs to have -2 since the matrix includes y values
        m_samples, n_features = X[i].shape
        standard_deviation_list = []
        for j in range(n_features - 1):
            summation = np.sum((X[i][:, j] - list(mean.values())[i][j]) ** 2)
            standard_deviation_list.append(np.sqrt((1 / (n_features - 2)) * summation))
        standard_deviation_dictionary[classes[i]] = standard_deviation_list
    return standard_deviation_dictionary


def probability_vector(_m_samples, _y_mean, _index, _normal_distribution):
    # create a vector with the product of feature's normal distribution and class probability for each sample
    X_normal_distribution = np.zeros((_m_samples, 1))
    for i in range(_m_samples):
        X_probability = list(_y_mean.values())[_index]
        for j in _normal_distribution[i]:
            X_probability *= j
        X_normal_distribution[i] = X_probability
    return X_normal_distribution


def normal_distribution_calculate(X, mean, standard_deviation, y_mean, classes):
    # create a dictionary for normal distribution
    probability_dictionary = {}
    m_samples, n_features = X.shape

    for i in range(len(classes)):
        index = i
        # create matrices for mean and standard deviation
        mean_matrix = np.zeros((m_samples, n_features))
        mean_list = list(mean.values())[index]
        standard_deviation_matrix = np.zeros((m_samples, n_features))
        standard_deviation_list = list(standard_deviation.values())[index]

        for j in range(m_samples):
            mean_matrix[j] = mean_list
            standard_deviation_matrix[j] = standard_deviation_list

        exponential = (-np.divide(np.square(X - mean_matrix), 2 * np.square(standard_deviation_matrix)))
        precursor = 1 / (np.sqrt(2 * np.pi) * standard_deviation_matrix)
        exponential = exponential.astype(float)
        normal_distribution = precursor * np.exp(exponential)

        probs = probability_vector(m_samples, y_mean, index, normal_distribution)
        probability_dictionary[index] = probs

    return probability_dictionary


def predict(y, y_true, classes):
    m_samples = len(list(y.values())[0])
    y_prediction_value = []
    # find the max value of each class per sample and create a vector of predictions
    for i in range(m_samples):
        sample_dictionary = {}
        for j in range(len(y)):
            sample_dictionary[j] = y[j][i]
        maximum = max(sample_dictionary, key=lambda x: sample_dictionary[x])
        y_prediction_value.append(maximum)
    y_prediction_name = []
    # convert predictions back into original names
    for i in y_prediction_value:
        for j in range(len(classes)):
            if i == j:
                y_prediction_name.append(classes[j])
    # calculate accuracy
    accuracy_value = 0
    for i in range(m_samples):
        if y_prediction_name[i] == y_true[i]:
            accuracy_value += 1
    accuracy_value = accuracy_value / m_samples

    return y_prediction_name, accuracy_value


if __name__ == "__main__":
    # import the data
    df = pd.read_csv("iris.csv")

    data_all = df.to_numpy()

    X_data = data_all[:, :-1]
    y_data = data_all[:, -1:]

    # splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=1)
    print(y_train.shape)
    update_data = np.column_stack((X_train, y_train))

    # separate classes
    data_all, classes_all = separate_classes(update_data)

    # find probabilities of class occurring
    y_mean_class = y_prob(data_all, X_train, classes_all)

    # find mean, standard deviation, and normal distribution
    X_mean = mean_calculate(data_all, classes_all)
    X_standard_deviation = standard_deviation_calculate(data_all, X_mean, classes_all)
    y_probability = normal_distribution_calculate(X_test, X_mean, X_standard_deviation, y_mean_class, classes_all)

    # predict
    prediction, accuracy = predict(y_probability, y_test, classes_all)

    print("prediction:\n", prediction)
    print("Accuracy: {}%".format(round(accuracy * 100, 2)))
