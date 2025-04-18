# Efficient-Classification-of-Electronic-Waste-using-Deep-Learning



This project utilizes deep learning techniques to classify electronic waste (e-waste) from mixed waste images. The system identifies e-waste items using a camera feed and displays the classification results via a web interface.

## Features

- **E-Waste Classification**: Detects and classifies e-waste from mixed waste using a deep learning model.
- **Web Interface**: Upload images for classification and view the results as e-waste or non-e-waste.
- **Real-Time Detection**: The system processes images and classifies waste items as they are uploaded.

## Requirements

To run this project, you need to install the following Python packages:

- `tensorflow` (for deep learning model training and inference)
- `opencv-python` (for video capture and image processing)
- `numpy` (for numerical operations)
- `pandas` (for handling data files)
- `flask` (for creating the web interface)Setup
Prepare the Dataset: Ensure you have a dataset of waste images, organized into folders such as train, valid, and test, each containing subfolders for e-waste and non-e-waste items.

Run the Training Script (ap.py): The first step is to run the ap.py script to begin training the model. This script will use the training and validation images to train the deep learning model.

bash
Copy
Edit
python ap.py
Model Files: After training, the model files (.pkl and .h5) will be generated. These files store the trained model's weights and other necessary information for classification.

Run the Web Interface Script (2.py): Once the model is trained and the files are generated, run the 2.py script to start the web interface for classification:

bash
Copy
Edit
python 2.py
Access the Web Interface: After running 2.py, a web link will be generated. Open this link in your browser to access the system’s interface for uploading and classifying images.

Usage
Upload Images: The web interface will allow you to upload images of waste items for classification.

Classify E-Waste or Non-E-Waste: After uploading, you will be prompted to classify the images:

E-Waste: Select this option to see the images identified as e-waste.

Non-E-Waste: Select this option to see the images identified as non-e-waste.

Notes
Make sure your camera and microphone are properly connected and functional if required by any feature.

You can adjust parameters in the code, such as the number of epochs for training or the batch size, for better performance.

Troubleshooting
Module Not Found Error: Ensure all required packages are installed using pip install -r requirements.txt.

Model Not Found: Ensure that the .pkl and .h5 files are generated after running ap.py.

Web Interface Issues: Make sure Flask is running and that no firewall or port conflicts are blocking the web interface.
- `keras` (for deep learning model support)
- `matplotlib` (for visualizations)

You can install these dependencies using `pip`. Create a `requirements.txt` file with the following content:

```bash
pip install -r requirements.txt

