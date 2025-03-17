import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import tensorflow as tf
import tempfile
import subprocess
import sys
import json
import uuid
import shutil
from pathlib import Path

from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "uploads")
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "visualizations")

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["VISUALIZATIONS_FOLDER"] = VISUALIZATIONS_FOLDER

# Allowed file extensions
ALLOWED_EXTENSIONS = {"pt", "pth", "onnx", "h5", "tflite", "pb"}

# Check file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")  # Frontend UI (upload form)

@app.route("/upload", methods=["POST"])
def upload_model():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        # Generate unique filename to prevent collisions
        unique_id = str(uuid.uuid4())[:8]
        orig_filename = file.filename
        filename = f"{unique_id}_{orig_filename}"
        file_ext = orig_filename.rsplit(".", 1)[1].lower()
        
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Process the uploaded model
        result = process_model(file_path, file_ext, unique_id)
        
        # Generate visualization
        vis_path = generate_visualization(file_path, file_ext, unique_id)
        if vis_path:
            result["visualization_path"] = vis_path
            
        return jsonify(result)

    return jsonify({"error": "Invalid file format"}), 400

def process_model(file_path, file_ext, unique_id):
    """Process the uploaded model based on its format."""
    try:
        model_info = {
            "file": os.path.basename(file_path), 
            "type": file_ext,
            "unique_id": unique_id
        }

        if file_ext in ["pt", "pth"]:  # PyTorch models
            try:
                # Try loading as a full model first
                model = torch.load(file_path, map_location="cpu")
                
                # Check if it's a state_dict or a full model
                if isinstance(model, dict):
                    # It's likely a state dictionary
                    model_info["architecture"] = "State Dictionary"
                    model_info["parameters"] = {
                        name: {"shape": tuple(param.shape), "params": param.numel()}
                        for name, param in model.items() if isinstance(param, torch.Tensor)
                    }
                    
                    # Save parameter names and shapes for visualization
                    structure = {}
                    for name, param in model.items():
                        if isinstance(param, torch.Tensor):
                            parts = name.split('.')
                            current = structure
                            for i, part in enumerate(parts):
                                if i == len(parts) - 1:
                                    current[part] = f"Tensor {tuple(param.shape)}"
                                else:
                                    if part not in current:
                                        current[part] = {}
                                    current = current[part]
                    model_info["structure"] = structure
                    
                elif hasattr(model, "named_parameters"):
                    # It's a proper model with parameters
                    model_info["architecture"] = str(model)
                    model_info["parameters"] = {
                        name: {"shape": tuple(param.shape), "params": param.numel()}
                        for name, param in model.named_parameters()
                    }
                    
                    # Generate structure visualization
                    structure = {}
                    for name, _ in model.named_parameters():
                        parts = name.split('.')
                        current = structure
                        for i, part in enumerate(parts):
                            if i == len(parts) - 1:
                                current[part] = "Parameter"
                            else:
                                if part not in current:
                                    current[part] = {}
                                current = current[part]
                    model_info["structure"] = structure
                else:
                    # Some other type of PyTorch object
                    model_info["architecture"] = f"PyTorch object of type: {type(model).__name__}"
                    model_info["details"] = str(model)
            except Exception as e:
                model_info["load_error"] = str(e)
                model_info["architecture"] = "Failed to analyze PyTorch model structure"

        elif file_ext == "onnx":  # ONNX models
            model = onnx.load(file_path)
            session = ort.InferenceSession(file_path)
            model_info["inputs"] = [
                {"name": inp.name, "shape": inp.shape, "type": inp.type} for inp in session.get_inputs()
            ]
            model_info["outputs"] = [
                {"name": out.name, "shape": out.shape, "type": out.type} for out in session.get_outputs()
            ]
            
            # Extract nodes for visualization
            nodes = []
            for node in model.graph.node:
                nodes.append({
                    "name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output)
                })
            model_info["nodes"] = nodes[:100]  # Limit to 100 nodes for display
            
        elif file_ext == "h5":  # Keras models
            model = load_model(file_path)
            layers = []
            for i, layer in enumerate(model.layers):
                layer_info = {
                    "name": layer.name,
                    "type": layer.__class__.__name__,
                    "output_shape": str(layer.output_shape),
                }
                
                # Get input/output connections
                if i > 0:
                    layer_info["inputs"] = [prev_layer.name for prev_layer in model.layers if hasattr(prev_layer, "output") and any(out in layer.input for out in [prev_layer.output])]
                
                layers.append(layer_info)
            
            model_info["layers"] = layers

        elif file_ext == "tflite":  # TensorFlow Lite models
            interpreter = tf.lite.Interpreter(model_path=file_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            model_info["inputs"] = [
                {"name": inp["name"], "shape": inp["shape"].tolist(), "dtype": str(inp["dtype"])} 
                for inp in input_details
            ]
            model_info["outputs"] = [
                {"name": out["name"], "shape": out["shape"].tolist(), "dtype": str(out["dtype"])} 
                for out in output_details
            ]
            
            # Get tensor details for visualization
            tensor_details = interpreter.get_tensor_details()
            model_info["tensors"] = [
                {"name": t["name"], "shape": t["shape"].tolist(), "dtype": str(t["dtype"]), "index": t["index"]}
                for t in tensor_details[:100]  # Limit to 100 tensors
            ]

        elif file_ext == "pb":  # TensorFlow frozen models
            try:
                # First try loading as SavedModel
                model = tf.saved_model.load(file_path)
                model_info["model_type"] = "SavedModel"
                model_info["signatures"] = {
                    key: {
                        "inputs": str(value.structured_input_signature),
                        "outputs": str(value.structured_outputs),
                    }
                    for key, value in model.signatures.items()
                }
            except Exception as e:
                # If that fails, try loading as frozen graph
                model_info["model_type"] = "Frozen Graph"
                model_info["load_error"] = f"Failed to load as SavedModel: {str(e)}"

        return model_info

    except Exception as e:
        return {"error": str(e)}

def generate_visualization(file_path, file_ext, unique_id):
    """Generate a visualization of the model and save it as HTML"""
    try:
        # Create visualization file path
        vis_filename = f"{unique_id}_visualization.html"
        vis_path = os.path.join(app.config["VISUALIZATIONS_FOLDER"], vis_filename)
        
        # Generate HTML visualization based on model type
        if file_ext in ["pt", "pth"]:
            generate_pytorch_visualization(file_path, vis_path)
        elif file_ext == "onnx":
            generate_onnx_visualization(file_path, vis_path)
        elif file_ext == "h5":
            generate_keras_visualization(file_path, vis_path)
        elif file_ext == "tflite":
            generate_tflite_visualization(file_path, vis_path)
        elif file_ext == "pb":
            generate_tensorflow_visualization(file_path, vis_path)
        
        # Return relative path for frontend
        return os.path.join("visualizations", vis_filename)
    except Exception as e:
        print(f"Visualization generation error: {str(e)}")
        return None

def generate_pytorch_visualization(model_path, output_path):
    """Generate visualization for PyTorch models"""
    try:
        model = torch.load(model_path, map_location="cpu")
        
        # Start building HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PyTorch Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .model-container { display: flex; flex-direction: column; }
                .layer { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
                .parameter { font-size: 12px; color: #555; }
                .tensor { background-color: #f0f8ff; }
                .module { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>PyTorch Model Visualization</h1>
            <div class="model-container">
        """
        
        # Handle state dict or model differently
        if isinstance(model, dict):
            html_content += "<h2>Model State Dictionary</h2>"
            for name, param in model.items():
                if isinstance(param, torch.Tensor):
                    shape_str = "×".join([str(s) for s in param.shape])
                    html_content += f"""
                    <div class="layer tensor">
                        <strong>{name}</strong> 
                        <div class="parameter">Shape: {shape_str}, Size: {param.numel()} parameters</div>
                    </div>
                    """
        elif hasattr(model, "named_modules"):
            html_content += "<h2>Model Architecture</h2>"
            # Add model type
            html_content += f"<p>Model type: {type(model).__name__}</p>"
            
            # Visualize modules
            for name, module in model.named_modules():
                if name == '':  # Skip the root module
                    continue
                    
                html_content += f"""
                <div class="layer module">
                    <strong>{name}</strong>: {module.__class__.__name__}
                    <div class="parameter">{str(module)}</div>
                </div>
                """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        # Create error visualization
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error Visualizing PyTorch Model</title></head>
            <body>
                <h1>Error Visualizing PyTorch Model</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """)

def generate_onnx_visualization(model_path, output_path):
    """Generate visualization for ONNX models"""
    try:
        model = onnx.load(model_path)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ONNX Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .node { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; background-color: #f5f5f5; }
                .inputs, .outputs { font-size: 12px; color: #555; }
                .info { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>ONNX Model Visualization</h1>
            <div class="info">
                <h2>Model Info</h2>
                <p><strong>IR Version:</strong> {}</p>
                <p><strong>Producer Name:</strong> {}</p>
                <p><strong>Domain:</strong> {}</p>
            </div>
            <h2>Model Graph</h2>
            <div class="nodes">
        """.format(
            model.ir_version,
            model.producer_name,
            model.domain if hasattr(model, 'domain') else 'Not specified'
        )
        
        # Add nodes
        for i, node in enumerate(model.graph.node):
            if i >= 100:  # Limit to first 100 nodes
                html_content += "<p>... (showing first 100 nodes only)</p>"
                break
                
            html_content += f"""
            <div class="node">
                <h3>{node.name if node.name else f"Node {i}"}</h3>
                <p><strong>Op Type:</strong> {node.op_type}</p>
                <div class="inputs">
                    <p><strong>Inputs:</strong></p>
                    <ul>
            """
            
            for inp in node.input:
                html_content += f"<li>{inp}</li>"
                
            html_content += """
                    </ul>
                </div>
                <div class="outputs">
                    <p><strong>Outputs:</strong></p>
                    <ul>
            """
            
            for out in node.output:
                html_content += f"<li>{out}</li>"
                
            html_content += """
                    </ul>
                </div>
            </div>
            """
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        # Create error visualization
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error Visualizing ONNX Model</title></head>
            <body>
                <h1>Error Visualizing ONNX Model</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """)

def generate_keras_visualization(model_path, output_path):
    """Generate visualization for Keras models"""
    try:
        model = load_model(model_path)
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Keras Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .layer { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; background-color: #f5f5f5; }
                .info { margin-top: 20px; }
                .model-summary { white-space: pre; font-family: monospace; background-color: #f8f8f8; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Keras Model Visualization</h1>
            <h2>Model Architecture</h2>
            <div class="layers">
        """
        
        # Add layers
        for i, layer in enumerate(model.layers):
            output_shape = str(layer.output_shape).replace('(', '[').replace(')', ']')
            
            html_content += f"""
            <div class="layer">
                <h3>Layer {i}: {layer.name}</h3>
                <p><strong>Type:</strong> {layer.__class__.__name__}</p>
                <p><strong>Output Shape:</strong> {output_shape}</p>
                <p><strong>Trainable:</strong> {layer.trainable}</p>
                <p><strong>Parameters:</strong> {layer.count_params()}</p>
            </div>
            """
        
        # Try to get model summary as string
        try:
            # Use StringIO to capture summary output
            import io
            from contextlib import redirect_stdout
            
            summary_io = io.StringIO()
            with redirect_stdout(summary_io):
                model.summary()
            summary_text = summary_io.getvalue()
            
            html_content += f"""
            <div class="info">
                <h2>Model Summary</h2>
                <div class="model-summary">{summary_text}</div>
            </div>
            """
        except:
            pass
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        # Create error visualization
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error Visualizing Keras Model</title></head>
            <body>
                <h1>Error Visualizing Keras Model</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """)

def generate_tflite_visualization(model_path, output_path):
    """Generate visualization for TFLite models"""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        tensor_details = interpreter.get_tensor_details()
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorFlow Lite Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .tensor { padding: 10px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; }
                .input { background-color: #e6f7ff; }
                .output { background-color: #f0fff0; }
                .intermediate { background-color: #f5f5f5; }
                .section { margin-top: 20px; }
            </style>
        </head>
        <body>
            <h1>TensorFlow Lite Model Visualization</h1>
            
            <div class="section">
                <h2>Input Tensors</h2>
        """
        
        # Add input details
        for inp in input_details:
            html_content += f"""
            <div class="tensor input">
                <h3>Input: {inp["name"]}</h3>
                <p><strong>Index:</strong> {inp["index"]}</p>
                <p><strong>Shape:</strong> {inp["shape"].tolist()}</p>
                <p><strong>Type:</strong> {inp["dtype"]}</p>
                <p><strong>Quantization:</strong> {inp.get("quantization", "None")}</p>
            </div>
            """
            
        html_content += """
            </div>
            
            <div class="section">
                <h2>Output Tensors</h2>
        """
        
        # Add output details
        for out in output_details:
            html_content += f"""
            <div class="tensor output">
                <h3>Output: {out["name"]}</h3>
                <p><strong>Index:</strong> {out["index"]}</p>
                <p><strong>Shape:</strong> {out["shape"].tolist()}</p>
                <p><strong>Type:</strong> {out["dtype"]}</p>
                <p><strong>Quantization:</strong> {out.get("quantization", "None")}</p>
            </div>
            """
            
        html_content += """
            </div>
            
            <div class="section">
                <h2>All Tensors</h2>
                <p>(Limited to first 50 tensors)</p>
        """
        
        # Add tensor details (limited)
        for i, tensor in enumerate(tensor_details):
            if i >= 50:  # Limit to first 50 tensors
                break
                
            tensor_type = "intermediate"
            if any(tensor["index"] == inp["index"] for inp in input_details):
                tensor_type = "input"
            elif any(tensor["index"] == out["index"] for out in output_details):
                tensor_type = "output"
                
            html_content += f"""
            <div class="tensor {tensor_type}">
                <h3>Tensor {i}: {tensor["name"]}</h3>
                <p><strong>Index:</strong> {tensor["index"]}</p>
                <p><strong>Shape:</strong> {tensor["shape"].tolist()}</p>
                <p><strong>Type:</strong> {tensor["dtype"]}</p>
            </div>
            """
            
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        # Create error visualization
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error Visualizing TFLite Model</title></head>
            <body>
                <h1>Error Visualizing TensorFlow Lite Model</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """)

def generate_tensorflow_visualization(model_path, output_path):
    """Generate visualization for TensorFlow models"""
    try:
        try:
            # Try loading as SavedModel
            model = tf.saved_model.load(model_path)
            is_saved_model = True
        except:
            is_saved_model = False
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TensorFlow Model Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .signature { padding: 15px; margin: 10px 0; border: 1px solid #ddd; border-radius: 5px; background-color: #f5f5f5; }
                .section { margin-top: 20px; }
                pre { background-color: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>TensorFlow Model Visualization</h1>
        """
        
        if is_saved_model:
            html_content += """
            <div class="section">
                <h2>SavedModel</h2>
            """
            
            # Add signatures if available
            if hasattr(model, 'signatures') and model.signatures:
                html_content += "<h3>Model Signatures</h3>"
                for key, value in model.signatures.items():
                    html_content += f"""
                    <div class="signature">
                        <h4>Signature: {key}</h4>
                        <h5>Inputs:</h5>
                        <pre>{str(value.structured_input_signature)}</pre>
                        <h5>Outputs:</h5>
                        <pre>{str(value.structured_outputs)}</pre>
                    </div>
                    """
            else:
                html_content += "<p>No signatures found in the model.</p>"
                
            # Try to get variable info if available
            try:
                variables = []
                for var in model.variables:
                    variables.append({
                        "name": var.name,
                        "shape": var.shape.as_list(),
                        "dtype": str(var.dtype)
                    })
                
                if variables:
                    html_content += """
                    <h3>Variables</h3>
                    <ul>
                    """
                    
                    for var in variables[:50]:  # Limit to 50 variables
                        html_content += f"""
                        <li>
                            <strong>{var["name"]}</strong> - Shape: {var["shape"]}, Type: {var["dtype"]}
                        </li>
                        """
                        
                    if len(variables) > 50:
                        html_content += f"<li>... and {len(variables) - 50} more variables</li>"
                        
                    html_content += "</ul>"
            except:
                pass
        else:
            html_content += """
            <div class="section">
                <h2>Frozen Graph / Other TensorFlow Format</h2>
                <p>This appears to be a TensorFlow frozen graph or another format.</p>
                <p>Limited visualization capabilities are available for this format.</p>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
    except Exception as e:
        # Create error visualization
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head><title>Error Visualizing TensorFlow Model</title></head>
            <body>
                <h1>Error Visualizing TensorFlow Model</h1>
                <p>{str(e)}</p>
            </body>
            </html>
            """)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/static/visualizations/<path:filename>')
def serve_visualization(filename):
    return send_from_directory(app.config['VISUALIZATIONS_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)