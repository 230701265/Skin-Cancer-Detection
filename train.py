import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gc  # Garbage collector

# Limit TensorFlow memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
PROCESSED_DIR = 'processed_data'
MODEL_DIR = 'model'
BATCH_SIZE = 16  # Reduced batch size for better learning
EPOCHS = 50  # Increased epochs
LEARNING_RATE = 0.0005  # Adjusted learning rate
IMG_SIZE = (128, 128)
MEMORY_EFFICIENT = True
TRAIN_BATCHES = 1
TEST_BATCHES = 1
SAMPLE_SIZE = 1000  # Increased sample size

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """Load a minimal subset of the preprocessed data"""
    print("Loading minimal dataset for quick testing...")
    
    # Load just a small subset of data
    try:
        # Load first batch only
        X_train = np.load(os.path.join(PROCESSED_DIR, 'train_images_batch0.npy'))[:SAMPLE_SIZE]
        y_train = np.load(os.path.join(PROCESSED_DIR, 'train_labels_batch0.npy'))[:SAMPLE_SIZE]
        X_test = np.load(os.path.join(PROCESSED_DIR, 'test_images_batch0.npy'))[:SAMPLE_SIZE//4]
        y_test = np.load(os.path.join(PROCESSED_DIR, 'test_labels_batch0.npy'))[:SAMPLE_SIZE//4]
        
        # Ensure images are in the correct shape
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        print(f"Training set: {X_train.shape}")
        print(f"Testing set: {X_test.shape}")
        
        # Load the label encoder
        le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        print(f"Number of classes: {len(le.classes_)}")
        
        # Convert labels to one-hot encoding
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(le.classes_))
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=len(le.classes_))
        
        return X_train, y_train, X_test, y_test, le, False
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, True

def build_model(input_shape, num_classes):
    """Build a CNN model for skin disease classification"""
    # Clear Keras backend
    K.clear_session()
    
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Fourth convolutional block
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        # Flatten and dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile with Adam optimizer
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(X_train, y_train, X_test, y_test, le):
    """Train the model with improved settings"""
    num_classes = len(le.classes_)
    input_shape = (*IMG_SIZE, 3)
    
    # Build model
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # Calculate class weights
    class_counts = np.sum(y_train, axis=0)
    total_samples = np.sum(class_counts)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # Changed to monitor accuracy
            patience=10,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',  # Changed to monitor accuracy
            factor=0.5,  # Less aggressive reduction
            patience=5,  # Increased patience
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, 'skin_cancer_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    # Data augmentation with more aggressive settings
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )
    
    # Train with data augmentation and class weights
    print("Starting training with data augmentation and class weights...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        class_weight=class_weights,  # Added class weights
        verbose=1
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, 'skin_cancer_model.h5'))
    
    # Evaluate model
    evaluate_model(model, X_test, y_test, le)
    
    # Plot training metrics
    plot_training_history(history)
    
    return model, history

def evaluate_model(model, X_test, y_test, le):
    """Evaluate the model and create classification reports"""
    # Predict test data
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Get unique classes in the test data
    unique_classes = np.unique(y_true)
    class_names = [le.classes_[i] for i in unique_classes]
    
    # Create classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=unique_classes,
        target_names=class_names,
        output_dict=True
    )
    
    # Print report
    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_pred,
        labels=unique_classes,
        target_names=class_names
    ))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations to the confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    return report

def plot_training_history(history):
    """Plot the training and validation metrics"""
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))

if __name__ == "__main__":
    print("Starting model training...")
    
    # Load preprocessed data
    try:
        X_train, y_train, X_test, y_test, le, use_batches = load_data()
        
        # Train the model
        if use_batches or MEMORY_EFFICIENT:
            print("Using memory-efficient batch training...")
            model, history = train_model(X_train, y_train, X_test, y_test, le)
        else:
            print("Using standard training with full dataset...")
            model, history = train_model(X_train, y_train, X_test, y_test, le)
        
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc() 
        traceback.print_exc() 