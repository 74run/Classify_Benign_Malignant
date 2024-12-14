import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('my_model_hybrid.h5')

# History of predictions (for comparison feature)
prediction_history = []

# Preprocess the image
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Predict function
def predict_image(img):
    # Preprocess and predict
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)

    # Determine result and confidence
    if prediction[0][0] > 0.5:
        result = "Malignant"
        confidence = prediction[0][0]
    else:
        result = "Benign"
        confidence = 1 - prediction[0][0]

    # Store prediction history
    prediction_history.append((result, confidence))
    return result, f"{confidence:.2f}"

# Dashboard performance metrics
def performance_dashboard():
    # Assume pre-saved images for the confusion matrix and classification report
    confusion_matrix_image_path = "confusion_matrix.png"
    classification_report_image_path = "report.png"
    
    return confusion_matrix_image_path, classification_report_image_path

# Function to show prediction history
def show_prediction_history():
    history_text = "\n".join([f"Result: {res}, Confidence: {conf:.2f}" for res, conf in prediction_history])
    return history_text if history_text else "No predictions yet."

# Create Gradio interface
def create_gradio_interface():
    # Define components for main interface
    image_input = gr.Image(type="pil", label="Upload Image")
    result_output = gr.Textbox(label="Prediction Result")
    confidence_output = gr.Textbox(label="Confidence Score")

    # Define components for dashboard
    confusion_matrix_output = gr.Image(label="Confusion Matrix")
    report_output = gr.Image(label="Classification Report")
    history_output = gr.Textbox(label="Prediction History", lines=5, interactive=False)

    # Define individual interfaces
    main_interface = gr.Interface(
        fn=predict_image,
        inputs=image_input,
        outputs=[result_output, confidence_output],
        title="Medical Image Classifier",
        description="Upload an image to classify it as Benign or Malignant.",
        live=True
    )

    dashboard_interface = gr.Interface(
        fn=performance_dashboard,
        inputs=[],
        outputs=[confusion_matrix_output, report_output],
        title="Performance Dashboard",
        description="View performance metrics and confusion matrix."
    )

    history_interface = gr.Interface(
        fn=show_prediction_history,
        inputs=[],
        outputs=history_output,
        title="Prediction History",
        description="View the history of predictions made during this session."
    )

    # Combine interfaces into tabs
    combined_interface = gr.TabbedInterface(
        [main_interface, dashboard_interface, history_interface],
        ["Classifier", "Dashboard", "History"]
    )
    combined_interface.launch()

# Run the Gradio interfaces
create_gradio_interface()
