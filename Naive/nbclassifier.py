from collections import defaultdict, Counter
import numpy as np
import math


class NBC:
    def y_occurs_no(self, matrix):
        samples = matrix.shape[0]   # number of samples in the dataset 'face+nonface'
        # Create a dictionary holding the probability of if a face occurs or a non-face
        face_dict = dict(Counter(matrix))
        for key in face_dict.keys():
            face_dict[key] = face_dict[key] / samples
        return face_dict


    def pixel_occurs_no(self, pix_list):
        samples = len(pix_list)     # number of samples in dataset by classification 0 or 1 not combined
        # holds a dict of key: value pairs 'pixel: numberofoccurances' in the column
        pix_dict = dict(Counter(pix_list))
        for key in pix_dict.keys():
            # take the log of(
            # from the key'pixel value(int)' add a 1 then devide by (samples + number of
            pix_dict[key] = math.log(key+1) / (samples+pix_dict[key])
        return pix_dict


    #get the max argument from the Y dictionary is face or not-face greater.
    def armax(self, y_dict):
        max_value_index = max(y_dict.values())
        max_key = [k for k, v in y_dict.items() if v == max_value_index]
        return max_key, max_value_index
    #create a dict with the values from the Y column.
    def naivebayes_dict(self):
        self.naive_dict = {}
        for value in self.vals:
            self.naive_dict[value] = defaultdict(list)      #default dictionary of type list


class NaiveBayesClassifier(NBC):
    def train(self, X, Y):
        # Get the unique Y values '1, 0' and create a defaultdict
        self.vals = np.unique(Y)
        self.naivebayes_dict()

        # dims of X
        x_rows, x_cols = np.shape(X)
        # Probabilities for y class to happen from data 1 = (number of y=1 /samples) 0 = (number of y=0 / samples)
        self.y_class_probabilities = self.y_occurs_no(Y)
        # For each class '0, 1' in our case calculate
        for value in self.vals:
            index = np.where(Y == value)[0]
            # Split the training class_x = class 0 or class 1 matrix
            class_x = X[index, :]
            new_x_rows, new_x_cols = np.shape(class_x)
            # collect the pixel values in each column and store them in the naivebayes dictionary by value 0'1
            for col in range(0, new_x_cols):
                self.naive_dict[value][col] += list(class_x[:, col])
        # for each Column inside of the naivebayes dictionary calculate the occurances of the feature value 'pixel' per
        # class 1'0
        for value in self.vals:
            for col in range(0, x_cols):
                self.naive_dict[value][col] = self.pixel_occurs_no(self.naive_dict[value][col])

    def classify_image(self, image):
        class_dict = {}
        # for each class loop adding to its respective class probabilities the values of pixel_values[image[pix]]
        for values in self.vals:
            y_class_probability = self.y_class_probabilities[values]
            for pix in range(0, len(image)):
                # for each pixel look up the pixels values
                pixel_values = self.naive_dict[values][pix]
                if image[pix] in pixel_values.keys():
                    y_class_probability += pixel_values[image[pix]]
                else:
                    y_class_probability *= 0
            # Store the sum of the y_class_probability for its respective class 1'0
            class_dict[values] = y_class_probability
        # get the argmax of the class 1'0 ' return which class has a higher value stored.
        return self.armax(class_dict)

    def image_classifer(self, X):
        self.y_predictions = []
        self.y_preds = []
        x_rows, x_cols = np.shape(X)
        # Each row is a image from X.  Classify the image and predict its value of face or non-face
        # store in y predictions
        for row in range(0, x_rows):
            image = X[row, :]
            prediction = self.classify_image(image)
            self.y_predictions.append(prediction[0])
            self.y_preds.append(prediction[1])
        return self.y_predictions, self.y_preds

