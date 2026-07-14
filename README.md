# SKU Manufacture & Accuracy Forecaster

An interactive AI web application built with Streamlit and TensorFlow/Keras. This application uses a Long Short-Term Memory (LSTM) Neural Network to generate future manufacturing quantity and accuracy forecasts based on historical SKU data.

## Features
- **Upload Dataset**: Direct CSV upload capability from the UI.
- **Dynamic Training**: Automatically trains a tailored LSTM model on the provided dataset.
- **Intelligent Forecasting**: Generates multi-step future forecasts for any specific SKU up to any target month.
- **Premium UI**: Clean, responsive, and visually appealing interface built with Streamlit and custom CSS.

## Getting Started

### Prerequisites
Make sure you have Python installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

To launch the Streamlit server, run the following command in the root directory:

```bash
python -m streamlit run app.py
```

Then, open your web browser and navigate to `http://localhost:8501`.

## Usage
1. Open the app in your browser.
2. Expand the sidebar and upload your historical `SKU_dataset.csv`.
3. Wait for the model to initialize and train.
4. Select a specific SKU from the dropdown menu.
5. Enter a target future month (e.g., `Jun-2026`).
6. Click **Forecast** to see the AI's predicted metrics!

## Project Structure
- `app.py`: The main Streamlit web application and UI layout.
- `model_utils.py`: Contains all the machine learning logic, data processing, and LSTM model architecture.
- `requirements.txt`: Python package dependencies.
