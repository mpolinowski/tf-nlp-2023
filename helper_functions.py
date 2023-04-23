import datetime
import itertools
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation, Rescaling, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness, RandomTranslation

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()


# evaluation functions
# accuracy, precision, recall and f1-score
def calculate_metrics(y_true, y_pred):
  """
  Calculate model accuracy, precision, recall and f1-score for binary classification models
  """

  # calculate accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100

  # calculate precision, recall and f1-score with weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

  model_results = {
    "accuracy": model_accuracy,
    "precision": model_precision,
    "recall": model_recall,
    "f1": model_f1
  }

  return model_results


# helper function to pre-process images for predictions
def prepare_image(file_name, img_height, img_width):
    # read in image
    img = tf.io.read_file(file_name)
    # image array => tensor
    img = tf.image.decode_image(img)
    # reshape to training size
    img = tf.image.resize(img, size=[img_height, img_width])
    # we don't need to normalize the image this is done by the model itself
    # img = img/255
    # add a dimension in front for batch size => shape=(1, 224, 224, 3)
    img = tf.expand_dims(img, axis=0)
    return img


# display random images from dir/class
def view_random_image(target_dir, target_class):
    target_folder = str(target_dir) + "/" + target_class
    random_image = random.sample(os.listdir(target_folder), 1)
    
    img = mpimg.imread(target_folder + "/" + random_image[0])
    plt.imshow(img)
    plt.title(str(target_class) + str(img.shape))
    plt.axis("off")
    
    return tf.constant(img)

# load and preprocess custom images
def load_and_preprocess_image(filename, img_shape, nomalize=True):
    # load image
    image = tf.io.read_file(filename)
    
    # decode image into tensor
    image = tf.io.decode_image(img, channels=3)
    
    # resize image
    tf.image.resize(image, [img_shape, img_shape])
    
    # models like efficientnet don't
    #  need normalization -> make it optional
    if normalize:
        return image/255
    else:
        return image
        
# create a callback to track experiments in TensorBoard
def create_tensorboard_callback(dir_name, experiment_name):
    # log progress to log directory
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    print(f"INFO :: Saving TensorBoard Log to: {log_dir}")
    return tensorboard_callback

# create a training checkpoint callback
def create_checkpoint_callback(dir_name, experiment_name):
    # log progress to log directory
    filepath = dir_name + "/" + experiment_name
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath, monitor='val_accuracy', verbose=0, save_best_only=True,
        save_weights_only=True, save_freq='epoch')
    print(f"INFO :: Saving Checkpoint to: {filepath}")
    return checkpoint_callback

# early stop callback
def create_early_stop_callback(monitor='val_loss',
                                  min_delta=0.0001,
                                  patience=10,
                                  restore_best_weights=True):
    early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, min_delta=min_delta,
            patience=patience, restore_best_weights=restore_best_weights)
    print(f"INFO :: Early-stop set to: min_delta {min_delta}")
    return early_stop_callback

# reduce learning rate callback
def create_reduce_learning_rate_callback(monitor="val_loss",  
                                        factor=0.2,
                                        patience=2,
                                        min_lr=1e-7):
    reduce_learning_rate_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",  
            factor=factor, # multiply the learning rate by 0.2 (reduce by 5x)
            patience=patience,
            verbose=1, # print out when learning rate goes down 
            min_lr=min_lr)
    print(f"INFO :: Reduce learning rate set to: min lr {min_lr}")
    return reduce_learning_rate_callback


# helper function to create a model
def create_model(model_url, num_classes, augmented):
    # download pre-trained model as a keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extractor_layer')
    
    # create sequential model
    model = tf.keras.Sequential([
        Rescaling(1./255, input_shape=IMG_SHAPE+(3,)),
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    return model


# helper function to create an augmented model
data_augmentation_layer_no_rescaling = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.2),
    RandomTranslation(
            height_factor=(-0.2, 0.3),
            width_factor=(-0.2, 0.3),
            fill_mode='reflect',
            interpolation='bilinear'),
    RandomContrast(0.2),
    RandomBrightness(0.2)
], name="data_augmentation")

# create the model
def create_augmented_model(model_url, num_classes):
    # download pre-trained model as a keras layer
    feature_extractor_layer = hub.KerasLayer(model_url,
                                             trainable=False,
                                             name='feature_extractor_layer')
    
    # create sequential model
    model = tf.keras.Sequential([
        Rescaling(1./255, input_shape=IMG_SHAPE+(3,)),
        data_augmentation_layer,
        feature_extractor_layer,
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])

    return model


# Plot the validation and training accuracy separately
def plot_accuracy_curves(history1, title1, history2, title2):
    accuracy1 = history1.history['accuracy']
    val_accuracy1 = history1.history['val_accuracy']
    epochs1 = range(len(history1.history['accuracy']))

    accuracy2 = history2.history['accuracy']
    val_accuracy2 = history2.history['val_accuracy']
    epochs2 = range(len(history2.history['accuracy']))

    # Plot accuracy
    plt.figure(figsize=(12, 12))
        
    plt.subplot(2, 2, 1)
    plt.plot(epochs1, accuracy1, label='training_accuracy')
    plt.plot(epochs1, val_accuracy1, label='val_accuracy')
    plt.title(title1)
    plt.xlabel('Epochs')
    plt.legend();

    plt.subplot(2, 2, 2)
    plt.plot(epochs2, accuracy2, label='training_accuracy')
    plt.plot(epochs2, val_accuracy2, label='val_accuracy')
    plt.title(title2)
    plt.xlabel('Epochs')
    plt.legend();


def combine_training_curves(original_history, new_history, pretraining_epochs):
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    print(len(acc))

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([pretraining_epochs-1, pretraining_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([pretraining_epochs-1, pretraining_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()



# making the process a bit more visually appealing
def predict_and_plot(model, file_name, class_names):
    # load the image and preprocess
    img = prepare_image(file_name)
    # run prediction
    prediction = model.predict(img)
    # get predicted class name
    pred_class = class_names[int(tf.round(prediction))]
    # plot image & prediction
    plt.imshow(mpimg.imread(file_name))
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# adapt plot function for multiclass predictions
def predict_and_plot_multi(model, file_name, class_names):
    # load the image and preprocess
    img = prepare_image(file_name)
    # run prediction
    prediction = model.predict(img)
    # check for multiclass
    if len(prediction[0]) > 1:
        # pick class with highest probability
        pred_class = class_names[tf.argmax(prediction[0])]
    else:
        # binary classifications only return 1 probability value
        pred_class = class_names[int(tf.round(prediction))]
    # plot image & prediction
    plt.imshow(mpimg.imread(file_name))
    plt.title(f"Prediction: {pred_class}")
    plt.axis(False)


# create the confusion matrix
def plot_confusion_matrix(y_pred, y_true, classes=None, figsize = (12, 12), text_size=7):
        cm = confusion_matrix(y_pred, y_true)

        # normalize
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # cm_norm
        # array([[1., 0.],
        #        [0., 1.]])

        number_of_classes = cm.shape[0]
        # 2

        # plot matrix
        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Greens)
        fig.colorbar(cax)

        if classes:
            labels = classes
        else:
            labels = np.arange(cm.shape[0])

        # axes lables
        ax.set(title="Confusion Matrix",
              xlabel="Prediction",
              ylabel="True",
              xticks=np.arange(number_of_classes),
              yticks=np.arange(number_of_classes),
              xticklabels=labels,
              yticklabels=labels)

        ax.xaxis.set_label_position("bottom")
        ax.title.set_size(20)
        ax.xaxis.label.set_size(20)
        ax.yaxis.label.set_size(20)
        ax.xaxis.tick_bottom()

        # vertical x-labels
        plt.xticks(rotation=70, fontsize=text_size)
        plt.yticks(fontsize=text_size)


        # colour threshold
        threshold = (cm.max() + cm.min()) / 2.

        # add text to cells
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i , f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
            horizontalalignment="center",
            color="white" if cm[i, j] > threshold else "black",
            size=text_size)


# function to pick a random image and run prediction
def random_image_prediction(model, images, true_labels, classes):
    # create random image index
    i = random.randint(0, len(images))
    # select label of image at index i
    true_label = classes[true_labels[i]]
    # pick corresponding image
    target_image = images[i]
    # reshape image and pass it to prediction
    pred_probabilities = model.predict(target_image.reshape(1, 28, 28))
    # select class with highest probability
    pred_label = classes[pred_probabilities.argmax()]
    
    # plot the b&w image
    plt.imshow(target_image, cmap=plt.cm.binary)
    
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"
        
    plt.xlabel("Prediction: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100*tf.reduce_max(pred_probabilities),
                                                     true_label), color = color)


# function to pick a random image and run prediction
def random_image_map_prediction(model, images, true_labels, classes):
    
    ran_gen = np.random.default_rng()
    
    plt.figure(figsize=(12, 12))
    
    for i in range(9):
        # select random image
        random_index = ran_gen.integers(low=0, high=len(images), size=1)
        target_image = images[random_index[0]]
        true_label = classes[true_labels[random_index[0]]]
        # reshape image and pass it to prediction
        pred_probabilities = model.predict(target_image.reshape(1, 28, 28))
        # select class with highest probability
        pred_label = classes[pred_probabilities.argmax()]
        
        ax = plt.subplot(3, 3, i+1)
        # plt.title(classes[train_labels[random_index[0]]])
    
        if pred_label == true_label:
            colour = "green"
            colourmap = "Greens"
        else:
            colour = "red"


# visualize predictions
# https://cs231n.github.io/neural-networks-case-study/

def decision_boundary(model, X, y):
    
    # define axis boundries for features and labels
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    
    # create meshgrid within boundries (fresh data to run predictions on)
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # stack both mesh arrays together
    x_in = np.c_[xx.ravel(), yy.ravel()]
    
    # make predictions using the trained model
    y_pred = model.predict(x_in)
    
    # check for multiclass-classification 
    if len(y_pred[0]) > 1:
        # reshape predictions
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        y_pred = np.round(y_pred).reshape(xx.shape)
        
    # plot decision boundry
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# measure prediction time to compare models
def time_to_prediction(model, samples):
    start_time = time.perf_counter()
    model.predict(samples)
    end_time = time.perf_counter()
    time_to_prediction = end_time - start_time
    prediction_time_weighted = time_to_prediction / len(samples)

    return time_to_prediction, prediction_time_weighted