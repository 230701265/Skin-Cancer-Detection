# Skin Disease Detection System

A deep learning-based system for detecting and classifying skin diseases using the HAM10000 dataset. This project uses TensorFlow for model training and Flask for the web interface.

## Project Structure

```
Skin-Disease-Detection/
├── Dataset/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   └── HAM10000_metadata.csv
├── model/
│   └── label_encoder.pkl
├── processed_data/
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── app.py
├── create_test_model.py
├── docker-compose.yml
├── Dockerfile
├── preprocess.py
├── requirements.txt
└── train.py
```

## Prerequisites

- Docker and Docker Compose
- Git
- At least 4GB of RAM available for Docker
- Sufficient disk space for the dataset and processed files

## Complete Setup Guide (Docker)

### 1. Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd Skin-Disease-Detection

# Clean up any existing containers
docker-compose down
```

### 2. Data Preprocessing (Required First Step)

The preprocessing step is necessary to:
- Prepare the dataset for training
- Create necessary directories
- Generate processed data files
- Split data into training and testing sets

```bash
# Run preprocessing
docker-compose run --rm skin-disease-app python preprocess.py
```

This step only needs to be run once unless you want to reprocess the data.

### 3. Model Training (Required Second Step)

The training step is necessary to:
- Build and train the CNN model
- Save the trained model for predictions
- Generate model evaluation metrics

```bash
# Train the model
docker-compose run --rm skin-disease-app python train.py
```

Training time depends on your system's resources. This step only needs to be rerun if you want to retrain the model.

### 4. Start the Web Application

```bash
# Start the web application
docker-compose up --build
```

The web interface will be available at `http://localhost:8080`

### Important Notes

1. **Port Configuration**:
   - The web application is accessible on port 8080
   - This is the only port you need to use to access the application
   - The internal container port (5000) is automatically mapped to 8080

2. **Training Configuration**:
   - Batch size is set to 16 for memory efficiency
   - Training runs for 10 epochs by default
   - These settings can be adjusted in train.py if needed

### 5. Development and Testing

For development purposes, you can create a test model:

```bash
# Create a test model
docker-compose run --rm skin-disease-app python create_test_model.py
```

### Important Notes

1. **Order of Operations**:
   - Preprocessing MUST be run before training
   - Training MUST be run before starting the web application
   - The web application requires a trained model to make predictions

2. **Container Management**:
   - Use `docker-compose down` to stop and remove containers
   - Use `docker ps` to check running containers
   - Only one instance should be running at a time

3. **Data Persistence**:
   - The `model/` directory persists trained models
   - The `processed_data/` directory stores preprocessed data
   - These directories are mounted as volumes in Docker

## Troubleshooting

1. **Memory Issues**:
   - If preprocessing fails, ensure Docker has enough memory allocated
   - The container is limited to 4GB of RAM by default

2. **Port Conflicts**:
   - The web application runs on port 8080
   - Ensure this port is available on your system

3. **Container Issues**:
   - If containers won't start, try `docker-compose down` and rebuild
   - Check Docker logs for specific error messages

4. **Model Loading Issues**:
   - Ensure preprocessing and training completed successfully
   - Check the model directory for required files

## Development Workflow

1. Make code changes
2. Rebuild and restart containers:
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## API Endpoints

- `GET /`: Home page with upload interface
- `POST /predict`: Endpoint for making predictions on uploaded images

## Development

To create a test model for development:
```bash
python create_test_model.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HAM10000 dataset for providing the skin lesion images
- TensorFlow and Keras for the deep learning framework
- Flask for the web interface 