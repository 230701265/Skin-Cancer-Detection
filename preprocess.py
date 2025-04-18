import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import shutil
import gc  # Garbage collector

# Constants
DATA_DIR = 'Dataset'
PROCESSED_DIR = 'processed_data'
IMG_SIZE = (128, 128)
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 500  # Process images in batches to save memory

# Create processed data directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs('model', exist_ok=True)

def preprocess_dataset():
    print("Loading metadata...")
    metadata = pd.read_csv(os.path.join(DATA_DIR, 'HAM10000_metadata.csv'))
    
    # Check for duplicates in the dataset
    print(f"Total entries in metadata: {len(metadata)}")
    print(f"Unique lesion_id count: {metadata['lesion_id'].nunique()}")
    print(f"Unique image_id count: {metadata['image_id'].nunique()}")
    
    # Display class distribution
    print("\nClass distribution:")
    class_counts = metadata['dx'].value_counts()
    print(class_counts)
    
    # Create a mapping from image_id to file path
    image_paths = {}
    
    # Search for images in part_1 directory
    part1_dir = os.path.join(DATA_DIR, 'HAM10000_images_part_1')
    for img_file in os.listdir(part1_dir):
        if img_file.endswith('.jpg'):
            image_id = os.path.splitext(img_file)[0]
            image_paths[image_id] = os.path.join(part1_dir, img_file)
    
    # Search for images in part_2 directory
    part2_dir = os.path.join(DATA_DIR, 'HAM10000_images_part_2')
    for img_file in os.listdir(part2_dir):
        if img_file.endswith('.jpg'):
            image_id = os.path.splitext(img_file)[0]
            image_paths[image_id] = os.path.join(part2_dir, img_file)
    
    print(f"Found {len(image_paths)} images")
    
    # Add file path to metadata
    metadata['path'] = metadata['image_id'].map(image_paths)
    
    # Drop rows with missing image files
    metadata = metadata.dropna(subset=['path'])
    print(f"After dropping missing images: {len(metadata)} entries")
    
    # Encode diagnostic labels
    le = LabelEncoder()
    metadata['label'] = le.fit_transform(metadata['dx'])
    
    # Save the label encoder for later use
    joblib.dump(le, os.path.join('model', 'label_encoder.pkl'))
    print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Choose a representative image for each lesion to avoid duplicates
    # Group by lesion_id and take the first image for each lesion
    metadata_unique = metadata.drop_duplicates(subset=['lesion_id'])
    print(f"Unique lesions: {len(metadata_unique)}")
    
    # Split the data into training and testing sets
    train_df, test_df = train_test_split(
        metadata_unique, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=metadata_unique['label']
    )
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")
    
    # Save the dataframes
    train_df.to_csv(os.path.join(PROCESSED_DIR, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, 'test.csv'), index=False)
    
    # Save batch info
    train_batches = (len(train_df) + BATCH_SIZE - 1) // BATCH_SIZE
    test_batches = (len(test_df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with open(os.path.join(PROCESSED_DIR, 'batch_info.txt'), 'w') as f:
        f.write(f"train_batches={train_batches}\n")
        f.write(f"test_batches={test_batches}\n")
    
    # Process the images in batches
    process_images_in_batches(train_df, 'train')
    process_images_in_batches(test_df, 'test')
    
    return train_df, test_df, le

def process_images_in_batches(df, subset):
    """Process images in small batches to save memory"""
    output_dir = os.path.join(PROCESSED_DIR, subset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual processed images
    print(f"Processing {subset} images...")
    
    # Process in batches
    num_samples = len(df)
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, num_samples)
        batch_df = df.iloc[start_idx:end_idx]
        
        X_batch = []
        y_batch = []
        
        print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx}:{end_idx})...")
        
        for idx, row in batch_df.iterrows():
            try:
                # Load and resize the image
                img = Image.open(row['path'])
                img = img.resize(IMG_SIZE)
                img_array = np.array(img) / 255.0  # Normalize to [0,1]
                
                # Save processed image
                img.save(os.path.join(output_dir, f"{row['image_id']}.jpg"))
                
                # Store image array and label
                X_batch.append(img_array)
                y_batch.append(row['label'])
                
            except Exception as e:
                print(f"Error processing image {row['image_id']}: {e}")
        
        # Convert batch to numpy array
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        
        # Save batch data
        batch_path = os.path.join(PROCESSED_DIR, f'{subset}_images_batch{batch_idx}.npy')
        labels_path = os.path.join(PROCESSED_DIR, f'{subset}_labels_batch{batch_idx}.npy')
        
        np.save(batch_path, X_batch)
        np.save(labels_path, y_batch)
        
        print(f"Saved batch {batch_idx+1}: {len(X_batch)} images, shape: {X_batch.shape}")
        
        # Free memory
        del X_batch, y_batch
        gc.collect()
    
    print(f"Completed processing {subset} images in {num_batches} batches")

def display_sample_images(df, le):
    """Display sample images from each class"""
    try:
        plt.figure(figsize=(15, 12))
        
        # For each class
        for i, class_name in enumerate(le.classes_):
            # Get images for this class
            class_df = df[df['dx'] == class_name].head(5)
            
            # Display up to 5 images
            for j, (_, row) in enumerate(class_df.iterrows()):
                if j >= 5:  # Limit to 5 images per class
                    break
                    
                img = Image.open(row['path'])
                plt.subplot(len(le.classes_), 5, i*5 + j + 1)
                plt.imshow(img)
                plt.title(f"{class_name} ({row['dx_type']})")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DIR, 'sample_images.png'))
        print("Sample images saved to processed_data/sample_images.png")
    except Exception as e:
        print(f"Error creating sample visualization: {e}")

if __name__ == "__main__":
    print("Starting preprocessing...")
    train_df, test_df, le = preprocess_dataset()
    
    # Display sample images from each class
    print("Creating sample images visualization...")
    display_sample_images(pd.concat([train_df, test_df]), le)
    
    print("Preprocessing complete!") 