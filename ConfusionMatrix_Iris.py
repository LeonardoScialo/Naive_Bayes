import numpy as np
import pandas as pd


def scoring_formulas(matrix, dimensions, name, index):
    true_false_dict = {'{}_tp'.format(name): 0,
                       '{}_fp'.format(name): 0,
                       '{}_fn'.format(name): 0,
                       '{}_tn'.format(name): 0}

    for i in range(dimensions):
        for j in range(dimensions):
            if i == index and j == index:
                true_false_dict['{}_tp'.format(name)] += matrix[i, j]
            elif i != index and j == index:
                true_false_dict['{}_fp'.format(name)] += matrix[i, j]
            elif i == index and j != index:
                true_false_dict['{}_fn'.format(name)] += matrix[i, j]
            elif i != index and j != index:
                true_false_dict['{}_tn'.format(name)] += matrix[i, j]
    tp = true_false_dict['{}_tp'.format(name)]
    fp = true_false_dict['{}_fp'.format(name)]
    fn = true_false_dict['{}_fn'.format(name)]
    tn = true_false_dict['{}_tn'.format(name)]

    accuracy = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


class ConfusionMatrix:

    def __init__(self, actual_results, predicted_results, classes):
        self.state_dict = {}

        self.actual_results = actual_results.flatten().tolist()
        self.predicted_results = predicted_results
        self.classes = classes

    def Calculate_Confusion_Matrix(self):
        # append dictionary for all class to class results and set equal to zero
        for i in self.classes:
            for j in self.classes:
                self.state_dict['{}_{}'.format(i, j)] = 0
        # compare actual and predicted results then append dictionary
        for self.actual_results, self.predicted_results in zip(self.actual_results, self.predicted_results):
            self.state_dict['{}_{}'.format(self.actual_results, self.predicted_results)] += 1

        matrix_dimensions = int(np.sqrt(len(self.state_dict)))

        matrix = np.array(list(self.state_dict.values())).reshape(matrix_dimensions, matrix_dimensions)

        name_list = []
        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_list = []

        for class_index, class_name in enumerate(self.classes):
            accuracy, precision, recall, f1_score = scoring_formulas(matrix, matrix_dimensions, class_name, class_index)
            name_list.append(class_name)
            accuracy_list.append('{}%'.format(round(accuracy * 100, 2)))
            precision_list.append('{}%'.format(round(precision * 100, 2)))
            recall_list.append('{}%'.format(round(recall * 100, 2)))
            f1_list.append('{}%'.format(round(f1_score * 100, 2)))

        formula_data = pd.DataFrame({'Accuracy': accuracy_list,
                                     'Precision': precision_list,
                                     'Recall': recall_list,
                                     'F1-score': f1_list}, index=name_list)
        return matrix, formula_data, matrix_dimensions
