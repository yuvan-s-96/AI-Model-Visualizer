# AI Model Visualizer

## Overview
AI Model Visualizer is a Flask-based web application that allows users to upload AI model files and visualize their structure. The application supports various model formats, including:

- PyTorch (`.pt`, `.pth`)
- ONNX (`.onnx`)
- Keras (`.h5`)
- TensorFlow Lite (`.tflite`)
- TensorFlow Frozen Graph (`.pb`)

## Features
- Upload AI model files for analysis
- Extract and display model structure, layers, inputs, and outputs
- Generate visualizations for better interpretability
- Supports multiple AI frameworks

## Installation
### Prerequisites
Ensure you have the following installed on your system:
- Python 3.7+
- Flask
- PyTorch
- ONNX & ONNX Runtime
- TensorFlow & Keras

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/ai-model-visualizer.git
   cd ai-model-visualizer
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the application:
   ```sh
   python app.py
   ```
4. Open a browser and go to `http://127.0.0.1:5000/`

## Usage
1. Navigate to the web interface.
2. Upload a supported AI model file.
3. View the extracted model details and visualizations.

## Project Structure
```
/ai-model-visualizer
│── static/
│   ├── uploads/            # Uploaded model files
│   ├── visualizations/     # Generated model visualizations
│── templates/
│   ├── index.html         # Frontend UI
│── app.py                 # Flask backend
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
```

## API Endpoints
- `/` - Renders the homepage.
- `/upload` - Handles model file uploads and processing.
- `/static/visualizations/<filename>` - Serves visualization files.

## License
This project is licensed under the MIT License.

## Contributions
Feel free to submit issues and pull requests to enhance the application!

## Contact
For questions or suggestions, please contact `your-email@example.com`.

