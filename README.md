# Advanced Possum Total Length Predictor

This is an advanced Streamlit application built to predict the total length of a possum based on various physical and demographic attributes. The app features multiple input methods, detailed data analysis, and interactive visualizations.

***

### Features

* **Manual Prediction**: Use sliders and dropdowns to input specific possum measurements and get an instant prediction.
* **CSV Upload**: Upload a CSV file with multiple possum records to get batch predictions.
* **Presets**: Use pre-defined possum profiles (e.g., "Young Male," "Adult Female") for quick predictions.
* **Data Analysis**: View key insights such as feature importance and the distribution of your predictions.
* **Interactive Visualizations**: Explore relationships between features with scatter matrices, 3D plots, time-series charts, and box plots using `Plotly`.
* **Prediction History**: All manual predictions are saved in a session history, which can be viewed or exported as a CSV file.

***

### Installation

1.  **Clone the repository** (if applicable) or save the Streamlit code to a file named `app.py`.
2.  **Install the required libraries**:
    ```bash
    pip install streamlit pandas numpy scikit-learn plotly seaborn matplotlib
    ```
    *Note: The model itself is not included. You must have a pre-trained model file named `best_possum_model.pkl` saved in the same directory as the app.py file.*

***

### Usage

1.  **Ensure you have a trained model**. The code expects a model file named `best_possum_model.pkl` created using a library like `scikit-learn`.
2.  **Run the application from your terminal**:
    ```bash
    streamlit run app.py
    ```
3.  The app will open in your web browser. You can then navigate through the tabs to make predictions and explore the data.

***

### Model and Dataset

* **Model**: The application uses a machine learning model, specifically a **Random Forest Regressor**, to predict the possum's total length.
* **Dataset**: The model is trained on a possum dataset (`possum.csv`) which includes 14 variables such as `hdlngth` (head length), `skullw` (skull width), `taill` (tail length), `age`, `sex`, and `Pop` (population group).
* **Variables Used for Prediction**: The model uses the following features: `sex`, `age`, `hdlngth`, `skullw`, `taill`, `footlgth`, `earconch`, `eye`, `chest`, `belly`, and `Pop_other` (a one-hot encoded variable for the population).
