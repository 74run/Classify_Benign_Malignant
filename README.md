### **README for Medical Image Classification Project**  

---

#### **Project Overview**  
This project is a deep learning-based solution for the binary classification of medical images to distinguish between *benign* and *malignant* cases. Using a dataset containing over 33,000 images across various anatomical sites, this implementation focuses on the *upper extremity* subset. Advanced techniques like data augmentation and transfer learning are employed to handle class imbalance and achieve high predictive accuracy.

The project includes a fully functional user interface deployed on **Hugging Face Spaces** for easy accessibility and interaction.

---

#### **Features**  
- **Dataset Handling**: Includes preprocessing, normalization, and augmentation of images.
- **Deep Learning Model**: Transfer learning using 3 layers for accurate predictions.
- **Performance Metrics**: Visualization of the confusion matrix, classification report, and loss/accuracy curves.
- **User Interface**: Interactive Gradio-based application for image input and prediction.
- **Deployment**: Fully deployed on Hugging Face Spaces for global access.
- **Dashboard**: Real-time performance metrics and historical prediction comparisons.

---

#### **Installation and Setup**  

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**  
   Ensure you have Python 3.7+ installed. Install required packages:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset Preparation**  
   - Download the dataset and organize it into `data/upper_extremity` folder.
   - Follow the format:  
     ```
     data/
       upper_extremity/
         benign/
         malignant/
     ```

4. **Run the Application**  
   - Train the model:  
     ```bash
     python train.py
     ```
   - Launch the Gradio Interface:  
     ```bash
     python app.py
     ```

5. **Access the Deployed Application**  
   - Access the live application on **Hugging Face Spaces**:  
     [Medical Image Classifier]((https://huggingface.co/spaces/74run/Medical_Image_Classifier)).

---

#### **Technical Specifications**  

- **Programming Language**: Python  
- **Libraries Used**:
  - TensorFlow/Keras for deep learning
  - Pandas and NumPy for data handling
  - Scikit-learn for metrics and evaluation
  - Gradio for UI and interaction
  - Matplotlib for visualizations

- **Key Files**:    
  - `app.py`: Runs the Gradio-based user interface.  
  - `requirements.txt`: Lists all required Python libraries.  

---

#### **Usage**  

1. **Input an Image**  
   Upload a medical image using the drag-and-drop interface.

2. **Get Prediction**  
   The model predicts whether the image is *benign* or *malignant* along with confidence scores.

3. **View Metrics**  
   Access the dashboard for real-time metrics, including confusion matrix and accuracy graphs.

---

#### **Performance**  

- Dataset Size: 4,963 images (4,852 benign, 111 malignant).  
- Augmented malignant samples to address class imbalance.  
- Achieved an accuracy of **94%** and F1-score of **0.93** with precision-recall analysis.  

---

#### **Troubleshooting**  

1. **Gradio Interface Doesn't Start**  
   - Ensure all dependencies are installed.  
   - Check for errors in `app.py` logs.

2. **Model Training Issues**  
   - Verify dataset folder structure.  
   - Check GPU availability if training is slow.  


---

#### **Future Enhancements**  

1. Increase dataset size by incorporating other anatomical sites.
2. Implement advanced techniques for better handling of class imbalance.
3. Add explainability features to visualize important regions in predictions.

---

#### **Contributors**  
- Tarun Janapati
- Saint Louis University 

---

#### **License**  
This project is licensed under the [MIT License](LICENSE).

---

#### **Acknowledgments**  
- Hugging Face Spaces for easy deployment.
- [Dataset Source]((https://challenge2020.isic-archive.com/)).  
