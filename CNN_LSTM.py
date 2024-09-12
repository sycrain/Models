import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from keras.callbacks import CSVLogger, Callback
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def load_data_from_folders(folder_paths):
    X_data = []
    Y_labels = []

    for label, folder in enumerate(folder_paths):
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder, filename)
                data = np.loadtxt(file_path)  
                X_data.append(data[:, 1]) 
                Y_labels.append(label)  

    X_data = np.array(X_data)
    Y_labels = np.array(Y_labels)
    return X_data, Y_labels

folder_paths = ['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil']

X, Y = load_data_from_folders(folder_paths)
X = X.reshape(-1, 500, 1) 

Y_onehot = np_utils.to_categorical(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.2, random_state=0)

def create_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, strides=2, input_shape=(500, 1), padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, strides=2, padding="same", activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_structure.png', show_shapes=True) 
    return model

class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

csv_logger = CSVLogger('training_log.csv', append=False)
loss_history = LossHistory()

model = create_model()

history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test),
                    callbacks=[csv_logger, loss_history])

with open('loss_and_accuracy.txt', 'w') as f:
    for i in range(len(loss_history.losses)):
        f.write(f"{i+1}, {loss_history.losses[i]}, {loss_history.accuracy[i]}\n")
        #f.write(f"Epoch {i + 1}, Loss: {loss_history.losses[i]}, Accuracy: {loss_history.accuracy[i]}\n")

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

accuracy = accuracy_score(Y_true, Y_pred_classes)
precision = precision_score(Y_true, Y_pred_classes, average='weighted')
recall = recall_score(Y_true, Y_pred_classes, average='weighted')
f1 = f1_score(Y_true, Y_pred_classes, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

cm = confusion_matrix(Y_true, Y_pred_classes)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    #plt.savefig('confusion_matrix.png')
    plt.savefig('confusion_matrix.png', dpi=1200, bbox_inches='tight')
    plt.show()

plot_confusion_matrix(cm, classes=['5-Methylcytosine', 'Adenine', 'Cytosine', 'Guanine', 'Thymine', 'Uracil'])
