import numpy as np
import cv2
import glob
import pathlib
import sklearn.metrics as metrics


def normalize(sample):
    max_val = max(sample)
    min_val = min(sample)
    
    normalized = np.zeros(len(sample))
    
    for i in range(len(sample)):
        normalized[i] = (sample[i] - min_val) / (max_val - min_val)
        
        
    return normalized

def translate(label):
    if label == "cane":
        # dog is 0
        return 0 
    
    elif label == "cavallo":
        # horse is 1
        return 1
    
    elif label == "elefante":
        # elephant is 2
        return 2
    
    elif label == "farfalla":
        # butterfly is 3
        return 3
    
    elif label == "gallina":
        # chicken is 4
        return 4
    
    elif label == "gatto":
        # cat is 5
        return 5
    
    elif label == "mucca":
        # cattle is 6
        return 6
    
    elif label == "pecora":
        # sheep is 7
        return 7
    
    elif label == "ragno":
        # spider is 8
        return 8
    
    elif label == "scoiattolo":
        # squirrel is 9
        return 9


def read_and_set():
    global whole_data
    global X_train
    global X_test
    global y_train
    global y_test
    
    whole_data = list()

    X_train = np.array(list())
    X_test = np.array(list())

    y_train = np.array(list())
    y_test = np.array(list())
    
    for path in pathlib.Path("raw-img").iterdir():


        for filename in glob.glob(str(path) +'/*'): 
            file_size = len(glob.glob(str(path) + '/*'))
            # skipping troubled image
            if filename.split("\\")[2] == "eb35b30729f1073ed1584d05fb1d4e9fe777ead218ac104497f5c97faeebb5bb_640.png":
                continue

            img = cv2.imread(filename)

            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(30,30))

            normalized_img = normalize(img.flatten())

            label_and_img = np.concatenate((np.array([translate(filename.split("\\")[1])]),normalized_img), axis = 0)
            whole_data.append(label_and_img)



    whole_data = np.array(whole_data)
    np.random.shuffle(whole_data)

    X_train = whole_data[:int(whole_data.shape[0]*80/100),1:]    
    y_train = whole_data[:int(whole_data.shape[0]*80/100),0]

    X_test = whole_data[int(whole_data.shape[0]*80/100):,1:]  
    y_test = whole_data[int(whole_data.shape[0]*80/100):,0]
    
    
class Layer:

    def __init__(self, input_numbers, neuron_numbers):
        self.input_numbers = input_numbers
        self.neuron_numbers = neuron_numbers
        self.weights = 0.01 * np.random.randn(input_numbers, neuron_numbers)
        self.biases = np.random.randn(1, neuron_numbers)

    def forward_propagation(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backpropagation(self, delta_values):
        self.delta_weights = np.dot(self.inputs.T, delta_values)
        self.delta_biases = np.sum(delta_values, axis=0, keepdims=True)
        self.delta_inputs = np.dot(delta_values, self.weights.T)
        return self.delta_inputs
    
class ReLU:

    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(inputs, 0)
        return self.output

    def backpropagation(self, delta_values):
        self.delta_inputs = delta_values.copy()
        self.delta_inputs[self.inputs <= 0] = 0
        return self.delta_inputs
    
def Softmax_forward_propagation(inputs):
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            if inputs[i][j] >= 710:
                inputs[i][j] = 709

    exponents = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exponents / np.sum(exponents, axis=1, keepdims=True)
    return probabilities

def negative_log_likelihood(predictions, y_true):

    samples = len(predictions)
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    correct_confidences = predictions[range(samples), y_true.astype(int)]
    sample_losses = -np.log(correct_confidences)

    return sample_losses

class Activation_Softmax_NegativeLogLikelihood():

    def forward_propagation(self, inputs, y_true):

        self.Z = Softmax_forward_propagation(inputs)
        return negative_log_likelihood(self.Z, y_true)

    def backpropagation(self, delta_values, y_true):

        samples = len(delta_values)
        self.delta_inputs = delta_values.copy()
        self.delta_inputs[range(samples), y_true.astype(int)] -= 1
        self.delta_inputs = self.delta_inputs / samples
        
def optimize(layer, alpha):
    layer.weights += -alpha * layer.delta_weights
    layer.biases += -alpha * layer.delta_biases
    
class Model:

    def __init__(self, X_train, y_train, X_test, y_test, number_of_hidden_layers, input_numbers, neuron_numbers):
        self.layers = []
        self.activation_ReLUs = []
        self.number_of_hidden_layers = number_of_hidden_layers

        if number_of_hidden_layers > 0:
            self.layers.append(Layer(input_numbers, neuron_numbers))
            self.activation_ReLUs.append(ReLU())

            for i in range(1, number_of_hidden_layers):
                self.layers.append(Layer(self.layers[i - 1].neuron_numbers, neuron_numbers))
                self.activation_ReLUs.append(ReLU())

            self.layers.append(Layer(neuron_numbers, 10))

        else:
            self.layers.append(Layer(input_numbers, 10))

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self, epochs, alpha, batch_size):

        activation_and_loss = Activation_Softmax_NegativeLogLikelihood()
        training_steps = len(self.X_train) // batch_size

        if training_steps * batch_size < len(self.X_train):
            training_steps += 1

        for epoch in range(epochs):

            total_acc_sum = 0
            total_acc_count = 0

            total_loss_sum = 0
            total_loss_count = 0

            for step in range(training_steps):
                X_batch = self.X_train[step * batch_size:(step + 1) * batch_size]
                y_batch = self.y_train[step * batch_size:(step + 1) * batch_size]

                outputs = []

                outputs.append(self.layers[0].forward_propagation(X_batch))

                if self.number_of_hidden_layers > 0:

                    outputs.append(self.activation_ReLUs[0].forward_propagation(outputs[0]))

                    for i in range(1, len(self.layers)):
                        outputs.append(self.layers[i].forward_propagation(outputs[len(outputs) - 1]))

                        if i != len(self.layers) - 1:

                            outputs.append(self.activation_ReLUs[i].forward_propagation(outputs[len(outputs) - 1]))

                sample_losses = activation_and_loss.forward_propagation(outputs[len(outputs) - 1], y_batch)
                predictions = np.argmax(activation_and_loss.Z, axis=1)

                # Backpropagation starts
                activation_and_loss.backpropagation(activation_and_loss.Z, y_batch)

                delta_outputs = []
                delta_outputs.append(activation_and_loss.delta_inputs)

                for j in range(len(self.layers) - 1, -1, -1):
                    if j == 0:
                        self.layers[j].backpropagation(delta_outputs[len(delta_outputs) - 1])

                    else:
                        delta_outputs.append(self.layers[j].backpropagation(delta_outputs[len(delta_outputs) - 1]))
                        delta_outputs.append(self.activation_ReLUs[j - 1].backpropagation(delta_outputs[len(delta_outputs) - 1]))

                for k in range(len(self.layers)):
                    optimize(self.layers[k], alpha)

                total_acc_sum += np.sum(predictions == y_batch)
                total_acc_count += len(predictions == y_batch)

                total_loss_sum += np.sum(sample_losses)
                total_loss_count += len(sample_losses)

            accuracy = total_acc_sum / total_acc_count
            loss = total_loss_sum / total_loss_count
            if not epoch % 100:
                print("epoch:", epoch+100, " accuracy:", round(accuracy, 3), " loss:", round(loss, 3))
    
    def save(self,weights_file,biases_file):
        with open(weights_file, 'wb') as f:
            for i in range(len(self.layers)):
                np.save(f, self.layers[i].weights)
                
        
        with open(biases_file,'wb') as f:
            for i in range(len(self.layers)):
                np.save(f, self.layers[i].biases)
                
            
    def load_and_set(self,weights_file,biases_file):
        self.layers = []
        with open(weights_file,'rb') as f:
            for i in range(len(self.layers)):
                self.layers[i].weights = np.load(f)
                
        with open(biases_file,'rb') as f:
            for i in range(len(self.layers)):
                self.layers[i].biases = np.load(f)
    
    def test(self):

        activation_and_loss = Activation_Softmax_NegativeLogLikelihood()

        outputs = []

        outputs.append(self.layers[0].forward_propagation(self.X_test))

        if self.number_of_hidden_layers > 0:

            outputs.append(self.activation_ReLUs[0].forward_propagation(outputs[0]))

            for i in range(1, len(self.layers)):
                outputs.append(self.layers[i].forward_propagation(outputs[len(outputs) - 1]))

                if i != len(self.layers) - 1:
                    outputs.append(self.activation_ReLUs[i].forward_propagation(outputs[len(outputs) - 1]))

        activation_and_loss.forward_propagation(outputs[len(outputs) - 1], self.y_test)
        predictions = np.argmax(activation_and_loss.Z, axis=1)
        
        conf_matrix = metrics.confusion_matrix(y_test,predictions)
        accuracy = metrics.accuracy_score(self.y_test.astype(int), predictions)
        precision = metrics.precision_score(self.y_test.astype(int), predictions, average='macro')
        recall = metrics.recall_score(self.y_test.astype(int), predictions, average='macro')
        F1 = metrics.f1_score(self.y_test.astype(int), predictions, average='macro')
    
        print("Confusion Matrix: ",conf_matrix,"\n")
        print("Test Accuracy:", round(accuracy, 3), "Precision:", round(precision, 3), "Recall:", round(recall, 3), "F1 Score:", round(F1, 3))

# Execution Example
# This model has one hidden layer, 900 nodes in input layer and 100 neurons in both hidden and output layer

read_and_set()
model = Model(X_train,y_train,X_test,y_test,2,900,100)

# Training Example

# Below code trains a model in 700 epochs, with 0.005 learning rate and batch size of 16

# Remove the comment lines for training(not necessary, we already have a trained model in npy files)

# model.train(700,0.005,16)
# model.save("weights.npy","biases.npy")

# Trained model is loaded
model.load_and_set("weights.npy","biases.npy")

# Testing
model.test()
    
