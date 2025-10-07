#!/usr/bin/env python3
"""
Silicone Defect Data Preparation Script for Google Colab
========================================================
Prepares silicone defect images and annotations for YOLO training in Colab
"""

import os
import shutil
import yaml
import zipfile
from pathlib import Path
from PIL import Image

# Configuration
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation
OUTPUT_ZIP = "/content/silicone_data.zip"

def create_dataset_structure():
    """Create YOLO dataset directory structure"""
    print("Creating dataset structure...")
    
    # Create main directories
    dataset_dir = Path("/content/silicone_training_data")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train and val directories
    for split in ['train', 'val']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset structure created at: {dataset_dir}")
    return dataset_dir

def convert_kitti_to_yolo(kitti_line, img_width, img_height):
    """Convert KITTI format annotation to YOLO format"""
    parts = kitti_line.strip().split()
    if len(parts) < 8:
        return None
    
    # KITTI format: class_name, truncated, occluded, alpha, x1, y1, x2, y2, ...
    class_name = parts[0]
    x1, y1, x2, y2 = map(float, parts[4:8])
    
    # Convert to YOLO format (normalized coordinates)
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = x2 - x1
    height = y2 - y1
    
    # Normalize coordinates
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    # Map class name to numeric class
    class_mapping = {'defect': 0}  # Single class for all defects
    
    if class_name in class_mapping:
        return f"{class_mapping[class_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    return None

def prepare_dataset_colab():
    """Main function to prepare the dataset for Colab"""
    print("=" * 60)
    print("SILICONE DEFECT DATA PREPARATION FOR COLAB")
    print("=" * 60)
    
    # Create dataset structure
    dataset_dir = create_dataset_structure()
    
    # Get all image files from uploaded data
    # Assuming images are uploaded to /content/images and labels to /content/labels
    images_dir = Path("/content/images")
    labels_dir = Path("/content/labels")
    
    if not images_dir.exists():
        print("❌ Images directory not found. Please upload images to /content/images")
        return None
    
    if not labels_dir.exists():
        print("❌ Labels directory not found. Please upload labels to /content/labels")
        return None
    
    image_files = list(images_dir.glob("*.bmp")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")
    
    if not image_files:
        raise ValueError(f"No images found in {images_dir}")
    
    # Split dataset (80% train, 20% val)
    split_idx = int(len(image_files) * TRAIN_SPLIT)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    def process_split(image_list, split_name):
        """Process images for a specific split"""
        print(f"\nProcessing {split_name} split...")
        
        for img_path in image_list:
            # Copy image
            dst_img = dataset_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Process annotation
            label_name = img_path.stem + '.txt'
            annotation_path = labels_dir / label_name
            
            if annotation_path.exists():
                # Get image dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Convert annotations
                dst_label = dataset_dir / split_name / 'labels' / label_name
                
                with open(annotation_path, 'r') as src_f, open(dst_label, 'w') as dst_f:
                    for line in src_f:
                        if line.strip():
                            yolo_line = convert_kitti_to_yolo(line, img_width, img_height)
                            if yolo_line:
                                dst_f.write(yolo_line + '\n')
                
                print(f"  ✓ {img_path.name} -> {split_name}")
            else:
                print(f"  ⚠ No annotation found for {img_path.name}")
    
    # Process both splits
    process_split(train_images, 'train')
    process_split(val_images, 'val')
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # Number of classes
        'names': ['defect']  # Class names
    }
    
    yaml_path = dataset_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"\n✅ Dataset preparation completed!")
    print(f"📁 Dataset location: {dataset_dir}")
    print(f"📄 Dataset YAML: {yaml_path}")
    print(f"🎯 Classes: {dataset_yaml['names']}")
    print(f"📊 Training images: {len(train_images)}")
    print(f"📊 Validation images: {len(val_images)}")
    
    # Create zip file for easy download
    print(f"\n📦 Creating zip file...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(dataset_dir.parent)
                zipf.write(file_path, arcname)
    
    print(f"✅ Zip file created: {OUTPUT_ZIP}")
    
    return dataset_dir, yaml_path

if __name__ == "__main__":
    try:
        dataset_dir, yaml_path = prepare_dataset_colab()
        print("\n" + "=" * 60)
        print("🎉 DATA PREPARATION SUCCESSFUL!")
        print("=" * 60)
        print(f"Next step: Run the training script")
        print(f"Dataset ready at: {yaml_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
