# Creating a function to visualize training results
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plots the training history for a given model's training process.

    Parameters:
    - history: History object from the training of a Keras model.
    """
    fig, axs = plt.subplots(2)

    # Plot training & validation accuracy values
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('Model accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('Model loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


# Creating a function to generate and threshold predictions
def generate_thresholded_predictions(model, data, threshold=0.5):
    """
    Generates predictions using a given model and thresholds the predictions.

    Parameters:
    - model: Trained Keras model.
    - data: Input data for prediction.
    - threshold: Threshold value for binary classification.

    Returns:
    - Thresholded predictions.
    """
    predictions = model.predict(data)
    return (predictions > threshold).astype(int)

