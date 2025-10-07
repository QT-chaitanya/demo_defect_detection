# Silicone Defect Detection Demo

A real-time silicone defect detection application with React frontend and Flask backend, powered by YOLOv8 deep learning model.

## 🎯 Features

- **Real-time Defect Detection**: Process silicone tube images with trained YOLOv8 model
- **Auto-loop Processing**: Automatically cycle through images every 3 seconds
- **Visual Status Indicators**: Blinking PASSED/FAILED indicators with counters
- **Folder Selection**: Select any folder containing images for batch processing
- **Image Shuffling**: Randomly shuffle images for realistic testing
- **Error Handling**: Skip failed images and log errors automatically
- **Responsive Design**: Clean, industrial-style interface optimized for quality control

## 📁 Project Structure

```
demo/
├── backend/
│   ├── app.py                 # Flask API server with image shuffling
│   ├── inference_service.py   # YOLOv8 model inference service
│   ├── requirements.txt      # Python dependencies
│   └── app.log               # Application logs
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── FolderSelector.jsx
│   │   │   ├── ImageViewer.jsx
│   │   │   └── StatusIndicator.jsx
│   │   ├── App.jsx          # Main app component
│   │   └── index.js         # Entry point
│   ├── public/
│   └── package.json         # Node.js dependencies
├── silicone_trained_model/   # Trained YOLOv8 model (46.5% mAP50)
│   ├── weights/
│   │   ├── best.pt          # Best trained model
│   │   └── last.pt          # Last training checkpoint
│   ├── dataset/             # Training/validation data
│   └── reports/             # Training reports and visualizations
├── SiliconImages/           # Sample silicone tube images
│   └── flattern/           # Flattened images for testing
├── siliconannotated/        # Annotated training data
├── train_clean_v2.py       # Clean training script
├── prepare_silicone_data.py # Data preparation script
└── README.md
```

## 🚀 Usage Options

This project can be used in two ways:

### Option 1: Google Colab Training + Local Demo
### Option 2: Complete Local Setup

---

## 🌐 Option 1: Google Colab Training + Local Demo

### Step 1: Train Model in Google Colab

1. **Go to Google Colab**: [colab.research.google.com](https://colab.research.google.com)

2. **Create New Notebook**: Click "New Notebook"

3. **Upload Training Data**:
   - Upload `silicone_data.zip` to Colab
   - The file will appear in `/content/silicone_data.zip`

4. **Run Training Code**:
   ```python
   # Install packages
   !pip install ultralytics opencv-python pillow pyyaml

   # Import libraries
   import os
   import yaml
   import shutil
   import zipfile
   from pathlib import Path
   from ultralytics import YOLO
   from PIL import Image

   # Extract data
   print("📦 Extracting silicone_data.zip...")
   with zipfile.ZipFile("/content/silicone_data.zip", 'r') as zip_ref:
       zip_ref.extractall("/content/")

   # Prepare dataset
   def prepare_dataset():
       dataset_dir = Path("/content/silicone_training_data")
       dataset_dir.mkdir(parents=True, exist_ok=True)
       
       for split in ['train', 'val']:
           (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
           (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
       
       images_dir = Path("/content/Silicone defects-20251006T115610Z-1-001/flattern")
       labels_dir = Path("/content/silicontube/default/label_2")
       
       image_files = list(images_dir.glob("*.bmp"))
       print(f"Found {len(image_files)} images")
       
       split_idx = int(len(image_files) * 0.8)
       train_images = image_files[:split_idx]
       val_images = image_files[split_idx:]
       
       def convert_kitti_to_yolo(kitti_line, img_width, img_height):
           parts = kitti_line.strip().split()
           if len(parts) < 8:
               return None
           
           class_name = parts[0]
           x1, y1, x2, y2 = map(float, parts[4:8])
           
           x_center = (x1 + x2) / 2.0
           y_center = (y1 + y2) / 2.0
           width = x2 - x1
           height = y2 - y1
           
           x_center /= img_width
           y_center /= img_height
           width /= img_width
           height /= img_height
           
           return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
       
       def process_split(image_list, split_name):
           print(f"Processing {split_name} split...")
           
           for img_path in image_list:
               dst_img = dataset_dir / split_name / 'images' / img_path.name
               shutil.copy2(img_path, dst_img)
               
               label_name = img_path.stem + '.txt'
               annotation_path = labels_dir / label_name
               
               if annotation_path.exists():
                   with Image.open(img_path) as img:
                       img_width, img_height = img.size
                   
                   dst_label = dataset_dir / split_name / 'labels' / label_name
                   
                   with open(annotation_path, 'r') as src_f, open(dst_label, 'w') as dst_f:
                       for line in src_f:
                           if line.strip():
                               yolo_line = convert_kitti_to_yolo(line, img_width, img_height)
                               if yolo_line:
                                   dst_f.write(yolo_line + '\n')
       
       process_split(train_images, 'train')
       process_split(val_images, 'val')
       
       dataset_yaml = {
           'path': str(dataset_dir.absolute()),
           'train': 'train/images',
           'val': 'val/images',
           'nc': 1,
           'names': ['defect']
       }
       
       yaml_path = dataset_dir / 'dataset.yaml'
       with open(yaml_path, 'w') as f:
           yaml.dump(dataset_yaml, f)
       
       return yaml_path

   # Train model
   def train_model():
       print("🚀 Starting training...")
       
       yaml_path = prepare_dataset()
       
       model = YOLO('yolov8m.pt')
       
       results = model.train(
           data=str(yaml_path),
           epochs=100,
           imgsz=640,
           batch=16,
           name="silicone_defect_detection",
           patience=20,
           save=True,
           plots=True,
           project="/content/silicone_model_output"
       )
       
       print("✅ Training completed!")
       
       # Create download package
       output_path = Path("/content/silicone_model_output")
       download_zip = Path("/content/silicone_trained_model.zip")
       
       with zipfile.ZipFile(download_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
           for file_path in output_path.rglob("*"):
               if file_path.is_file():
                   arcname = file_path.relative_to(output_path)
                   zipf.write(file_path, arcname)
       
       print(f"📦 Download package created: {download_zip}")

   # Run training
   train_model()
   ```

5. **Download Trained Model**:
   ```python
   from google.colab import files
   files.download("/content/silicone_trained_model.zip")
   ```

### Step 2: Setup Local Demo

1. **Extract Downloaded Model**:
   ```bash
   unzip silicone_trained_model.zip
   ```

2. **Update Model Path** (if needed):
   ```bash
   nano backend/inference_service.py
   ```
   Ensure line 17 points to your extracted model:
   ```python
   self.model_path = "/path/to/extracted/silicone_model_output/weights/best.pt"
   ```

3. **Install Dependencies**:
   ```bash
   # Backend dependencies
   cd backend
   pip3 install -r requirements.txt
   
   # Frontend dependencies
   cd ../frontend
   npm install
   ```

4. **Start the Application**:
   ```bash
   # Terminal 1 - Backend
   cd backend
   python3 app.py
   
   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

5. **Access Demo**: Open `http://localhost:3000`

---

## 💻 Option 2: Complete Local Setup

### Prerequisites

- **Python 3.10+**
- **Node.js 16+**
- **GPU recommended** (for training, CPU works but slower)

### Step 1: Prepare Training Data

1. **Organize your data**:
   ```
   SiliconImages/flattern/     # Your silicone tube images (.bmp)
   siliconannotated/           # Your annotation files (.txt)
   ```

2. **Run data preparation**:
   ```bash
   python3 prepare_silicone_data.py
   ```

### Step 2: Train Model Locally

1. **Install training dependencies**:
   ```bash
   pip3 install ultralytics opencv-python pillow pyyaml
   ```

2. **Run training**:
   ```bash
   python3 train_clean_v2.py
   ```

3. **Expected results**:
   - Training time: 30-60 minutes (GPU) or 2-4 hours (CPU)
   - Model accuracy: ~46.5% mAP50
   - Output: `silicone_model_output/weights/best.pt`

### Step 3: Setup Demo Application

1. **Install dependencies**:
   ```bash
   # Backend
   cd backend
   pip3 install -r requirements.txt
   
   # Frontend
   cd ../frontend
   npm install
   ```

2. **Configure model path**:
   ```bash
   nano backend/inference_service.py
   ```
   Update line 17:
   ```python
   self.model_path = "/home/qualitas/Desktop/demo/silicone_model_output/weights/best.pt"
   ```

3. **Start application**:
   ```bash
   # Terminal 1 - Backend
   cd backend
   python3 app.py
   
   # Terminal 2 - Frontend
   cd frontend
   npm start
   ```

4. **Access demo**: `http://localhost:3000`

---

## 🎮 How to Use the Demo

### Basic Usage

1. **Select Image Folder**:
   - Enter path: `/home/qualitas/Desktop/demo/SiliconImages/flattern`
   - Click "Select Folder"
   - Images are automatically shuffled for realistic testing

2. **Start Processing**:
   - Click "Start Loop" for automatic processing (3-second intervals)
   - Use "Previous"/"Next" for manual navigation
   - Watch PASSED/FAILED counters update in real-time

3. **Monitor Results**:
   - **Original Image**: Shows input silicone tube image
   - **Detection Result**: Shows annotated image with detected defects
   - **Status Indicators**: Blinking PASSED/FAILED with counts

### Advanced Features

- **Image Shuffling**: Images are randomly shuffled for realistic testing
- **Confidence Threshold**: Adjustable in `backend/inference_service.py`
- **Error Handling**: Failed images are automatically skipped
- **Logging**: All activities logged to `backend/app.log`

## ⚙️ Configuration

### Model Settings

**Model Path** (`backend/inference_service.py`):
```python
self.model_path = "/home/qualitas/Desktop/demo/silicone_trained_model/weights/best.pt"
```

**Confidence Threshold** (`backend/inference_service.py`):
```python
self.confidence_threshold = 0.4  # Adjust sensitivity (0.1-0.9)
```

### Performance Tuning

- **Lower threshold (0.1-0.2)**: More sensitive, detects more defects
- **Higher threshold (0.4-0.6)**: Less sensitive, fewer false positives
- **Recommended**: 0.4 for production use

## 📊 Model Performance

- **Training Dataset**: 31 silicone tube images
- **Validation Accuracy**: 46.5% mAP50
- **Model Size**: ~52MB
- **Inference Speed**: ~15ms per image (GPU)
- **Supported Formats**: BMP, JPG, PNG, TIFF

## 🔧 Troubleshooting

### Common Issues

1. **Model not found**:
   ```bash
   # Check if model exists
   ls -la silicone_trained_model/weights/best.pt
   
   # Update path in inference_service.py
   nano backend/inference_service.py
   ```

2. **Port conflicts**:
   ```bash
   # Backend port (default: 5000)
   nano backend/app.py
   
   # Frontend port (default: 3000)
   nano frontend/package.json
   ```

3. **Low detection rate**:
   ```bash
   # Lower confidence threshold
   nano backend/inference_service.py
   # Change: self.confidence_threshold = 0.1
   ```

4. **Training errors**:
   ```bash
   # Check logs
   tail -f backend/app.log
   
   # Verify data format
   python3 prepare_silicone_data.py
   ```

### Performance Optimization

- **GPU Training**: Use CUDA-enabled PyTorch for faster training
- **Memory Issues**: Reduce batch size in training script
- **Slow Inference**: Enable GPU acceleration in inference service

## 📈 Expected Results

### Training Output
- **Training Time**: 15-30 minutes (Colab GPU) or 2-4 hours (local CPU)
- **Model Accuracy**: 46.5% mAP50 (excellent for small dataset)
- **Detection Rate**: 6/7 validation images show defects (at 0.1 threshold)

### Demo Performance
- **Processing Speed**: ~3 seconds per image (including display)
- **Accuracy**: Detects silicone defects with 46.5% precision
- **Reliability**: Handles various image sizes and formats

## 🎯 Use Cases

- **Quality Control**: Automated silicone tube defect detection
- **Production Monitoring**: Real-time defect screening
- **Training Demo**: Showcase AI-powered quality inspection
- **Research**: Base model for further defect detection research

## 📝 API Endpoints

- `POST /api/select_folder` - Select image folder (with shuffling)
- `POST /api/process_image` - Process single image for defects
- `GET /api/get_image/<path>` - Serve image files
- `GET /api/health` - Health check

## 🔄 Updates and Improvements

### Recent Updates
- ✅ Image shuffling for realistic testing
- ✅ Improved confidence threshold handling
- ✅ Better error handling and logging
- ✅ Optimized training scripts for Colab

### Future Enhancements
- 🔄 Multi-class defect detection (bubbles, cuts, particles)
- 🔄 Real-time video processing
- 🔄 Database integration for defect tracking
- 🔄 Mobile app interface

---

## 📞 Support

For issues or questions:
1. Check `backend/app.log` for error details
2. Verify model path and confidence threshold
3. Ensure all dependencies are installed
4. Check image format compatibility

**Happy Defect Detection!** 🎉