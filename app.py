import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load the pre-trained model
model = load_model('my_model.h5')

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
    if prediction[0] > 0.5:
        result = "Malignant"
        confidence = prediction[0][0]
    else:
        result = "Benign"
        confidence = 1 - prediction[0][0]

    # Store prediction history
    prediction_history.append((result, confidence))
    return result, f"{confidence:.2f}"

# Display metrics (dashboard)
def performance_dashboard():
    # Example ground truth and predictions (update with actual data in practice)
    y_true = [0, 0, 1, 1, 1, 0, 1]  # True labels (for demonstration)
    y_pred = [0, 1, 1, 1, 0, 0, 1]  # Predicted labels (for demonstration)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    # Save confusion matrix plot as an image
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"])
    
    # Return the confusion matrix image and the classification report text
    return "confusion_matrix.png", report

# Create Gradio interface
def create_gradio_interface():
    # Define components
    image_input = gr.Image(type="pil", label="Upload Image")
    result_output = gr.Textbox(label="Prediction Result")
    confidence_output = gr.Textbox(label="Confidence Score")
    dashboard_button = gr.Button("Show Performance Dashboard")  # Fixed the label issue
    confusion_matrix_output = gr.Image(label="Confusion Matrix")
    report_output = gr.Textbox(label="Classification Report")
    history_output = gr.Textbox(label="Prediction History")

    # Update dashboard on button click
    def show_dashboard():
        confusion_img, report = performance_dashboard()
        return confusion_img, report

    def show_prediction_history():
        history_text = "\n".join([f"Result: {res}, Confidence: {conf:.2f}" for res, conf in prediction_history])
        return history_text

    # Create main interface
    main_interface = gr.Interface(
        fn=predict_image, 
        inputs=image_input, 
        outputs=[result_output, confidence_output],
        title="Medical Image Classifier",
        description="Upload an image to classify it as Benign or Malignant.",
        live=True
    )

    # Create dashboard interface
    dashboard_interface = gr.Interface(
        fn=show_dashboard,
        inputs=[],
        outputs=[confusion_matrix_output, report_output],
        title="Performance Dashboard",
        description="View performance metrics and confusion matrix."
    )

    # Combine interfaces
    combined_interface = gr.TabbedInterface([main_interface, dashboard_interface], ["Classifier", "Dashboard"])
    combined_interface.launch()

# Run the Gradio interfaces
create_gradio_interface()
