#!/usr/bin/env python3
"""
Clean Training Script for Silicone Defect Detection - Google Colab Version
- Single output folder: /content/silicone_model_output
- Only best.pt and last.pt weights (overwrite when better)
- Reports only at the end
- No intermediate folders or cleanup needed
"""

import os
import yaml
import shutil
import zipfile
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# --- Configuration Section ---
train_images_path = "/content/Silicone defects-20251006T115610Z-1-001/flattern"
train_labels_path = "/content/silicontube/default/label_2"
epochs = 100  # Reduced for smaller dataset
batch_size = 16  # Higher for Colab GPU
img_size = 640
output_dir = "/content/silicone_model_output"
# --- End Configuration Section ---

def extract_data_from_zip():
    """Extract silicone_data.zip if not already extracted"""
    zip_path = Path("/content/silicone_data.zip")
    if zip_path.exists() and not Path("/content/Silicone defects-20251006T115610Z-1-001").exists():
        print("📦 Extracting silicone_data.zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("/content/")
        print("✅ Extraction completed!")
    else:
        print("✅ Data already extracted or zip not found")

def convert_kitti_to_yolo(bbox, img_width, img_height):
    """Convert KITTI format bbox to YOLO format"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

def create_dataset_yaml():
    """Create dataset structure and YAML file"""
    dataset_dir = Path(output_dir) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Create train and val directories
    for split in ['train', 'val']:
        (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = list(Path(train_images_path).glob("*.bmp")) + \
                  list(Path(train_images_path).glob("*.jpg")) + \
                  list(Path(train_images_path).glob("*.png"))
    
    if not image_files:
        raise ValueError(f"No images found in {train_images_path}")
    
    print(f"Found {len(image_files)} images")
    
    # Split dataset (80% train, 20% val)
    split_idx = int(len(image_files) * 0.8)
    train_images = image_files[:split_idx]
    val_images = image_files[split_idx:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Detect classes from label files
    class_mapping = {}
    class_counter = 0
    
    for label_file in Path(train_labels_path).glob("*.txt"):
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    class_name = parts[0]
                    if class_name not in class_mapping:
                        class_mapping[class_name] = class_counter
                        class_counter += 1

    def copy_and_convert_files(image_list, split):
        """Copy images and convert labels from KITTI to YOLO format"""
        print(f"Processing {split} split...")
        
        for img_path in image_list:
            # Copy image
            dst_img = dataset_dir / split / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Convert and copy label
            label_name = img_path.stem + '.txt'
            label_path = Path(train_labels_path) / label_name
            
            if label_path.exists():
                dst_label = dataset_dir / split / 'labels' / label_name
                
                # Get image dimensions
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Convert labels
                with open(label_path, 'r') as src_f, open(dst_label, 'w') as dst_f:
                    for line in src_f:
                        if line.strip():
                            parts = line.strip().split()
                            class_name = parts[0]
                            numeric_class = class_mapping[class_name]
                            
                            try:
                                # KITTI format: class_name, truncated, occluded, alpha, x1, y1, x2, y2, ...
                                x1, y1, x2, y2 = map(float, parts[4:8])
                                x_center, y_center, width, height = convert_kitti_to_yolo(
                                    [x1, y1, x2, y2], img_width, img_height)
                                dst_f.write(f"{numeric_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            except ValueError:
                                print(f"Warning: Invalid bbox in {label_path.name}: {line.strip()}")
            else:
                print(f"Warning: No label found for {img_path.name}")
    
    # Process train and validation sets
    copy_and_convert_files(train_images, 'train')
    copy_and_convert_files(val_images, 'val')
    
    # Create class names list
    class_names = [k for k, v in sorted(class_mapping.items(), key=lambda item: item[1])]
    
    print(f"Detected classes: {list(class_mapping.values())}")
    print(f"Class names: {class_names}")
    
    # Create dataset YAML
    dataset_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dataset_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    print(f"Dataset YAML created at: {yaml_path}")
    return yaml_path

def train_model():
    """Main training function"""
    print("==================================================")
    print("Starting SILICONE DEFECT training process...")
    print(f"Images: {train_images_path}")
    print(f"Labels: {train_labels_path}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Output: {output_dir}")
    print("💾 Weights: Only best.pt and last.pt (overwrite when better)")
    print("📊 Reports: Only at the end")
    print("==================================================")

    # Extract data first
    extract_data_from_zip()

    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    weights_dir = output_path / "weights"
    reports_dir = output_path / "reports"
    weights_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    # Create dataset
    print("Creating dataset structure...")
    dataset_yaml_path = create_dataset_yaml()
    
    # Load YOLOv8 MEDIUM model
    print("Loading YOLOv8 MEDIUM model...")
    model = YOLO('yolov8m.pt')
    
    print("Starting training...")
    results = model.train(
        data=str(dataset_yaml_path),
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        name="silicone_defect_detection",
        patience=20,
        save=True,
        plots=True,  # Enable plots for Colab
        project=str(output_path),
        # Memory-optimized augmentation parameters
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        mosaic=1.0, mixup=0.1, copy_paste=0.1,
        optimizer='AdamW', lr0=0.001, lrf=0.01,
        cos_lr=True,
        label_smoothing=0.05,
        dropout=0.05,
        box=7.5, cls=0.5, dfl=1.5,
        multi_scale=False,
        close_mosaic=10,
        save_json=True,
        workers=2,  # Reduced for Colab
        cache=True,  # Enable caching for Colab
        amp=True,
    )
    
    # Copy weights to our weights folder
    run_dir = Path(model.trainer.save_dir)
    final_best_weight = run_dir / "weights" / "best.pt"
    final_last_weight = run_dir / "weights" / "last.pt"

    if final_best_weight.exists():
        shutil.copy(final_best_weight, weights_dir / "best.pt")
        print(f"✅ Best model saved to: {weights_dir / 'best.pt'}")
    else:
        print(f"❌ Warning: best.pt not found at {final_best_weight}")

    if final_last_weight.exists():
        shutil.copy(final_last_weight, weights_dir / "last.pt")
        print(f"✅ Last model saved to: {weights_dir / 'last.pt'}")
    else:
        print(f"❌ Warning: last.pt not found at {final_last_weight}")

    # Generate reports at the end
    print("Generating final reports...")
    try:
        # Generate plots and reports
        val_results = model.val(plots=True, save_json=True)
        
        print(f"\n📈 Final Validation Results:")
        print(f"   mAP50: {val_results.box.map50:.3f}")
        print(f"   mAP50-95: {val_results.box.map:.3f}")
        
        # Copy reports from Ultralytics run to our reports folder
        for item in run_dir.iterdir():
            if item.is_file() and item.suffix in ['.png', '.jpg', '.csv', '.json']:
                shutil.copy(item, reports_dir / item.name)
                print(f"✅ Report saved to: {reports_dir / item.name}")
            elif item.is_dir() and item.name == "plots":
                for plot_file in item.iterdir():
                    shutil.copy(plot_file, reports_dir / plot_file.name)
                    print(f"✅ Report saved to: {reports_dir / plot_file.name}")
    except Exception as e:
        print(f"Warning: Could not generate reports: {e}")

    # Create download package
    print("\n📦 Creating download package...")
    download_zip = Path("/content/silicone_trained_model.zip")
    
    with zipfile.ZipFile(download_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_path.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(output_path)
                zipf.write(file_path, arcname)
    
    print(f"✅ Download package created: {download_zip}")

    print("==================================================")
    print("🎉 SILICONE DEFECT Training completed!")
    print(f"📁 Weights saved in: {weights_dir}/")
    print(f"📊 Reports saved in: {reports_dir}/")
    print(f"📦 Download package: {download_zip}")
    print("==================================================")

if __name__ == "__main__":
    train_model()
