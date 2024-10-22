# Time Series Forecasting API - Data Team

### Project Overview

This project is part of a **Time Series Forecasting Task**, which involves developing a machine learning model capable of making single-step predictions based on input time series data. The API accepts a dataset identifier and a list of timestamp-value pairs, processes the data, and returns a prediction.

### Key Features
- **Time Series Forecasting**: The model provides a forecast for the next time step based on the input time series data.
- **Feature Extraction**: Traditional machine learning models are used in combination with feature extraction techniques to enhance the accuracy of predictions.

### Directory Structure
```
├── LICENSE
├── README.md
├── TS_API.postman_collection.json   # Collection for testing the API with Postman
├── min_entries_inference_dataset_with_ids.csv   # CSV file with dataset details
├── requirements.txt   # Dependencies required to run the project
├── server.py   # Flask application hosting the API
├── ts-notebook.ipynb  # Pipeline work
├── models             # Saved models directory (instructions below)
├── Task.md            # Task description
```

### Requirements
To install and run the project, ensure you have the following dependencies:
- **Python 3.8+**
- **Flask** - Micro web framework for serving the API
- **Machine Learning Libraries**:
  - scikit-learn 1.2.2
  - pandas
  - numpy

These dependencies are included in the `requirements.txt` file. You can install them by running:

```bash
pip install -r requirements.txt
```

### Downloading and Setting Up the Saved Models

1. Visit the following link to download the pre-trained models: [Saved Models - Kaggle](https://www.kaggle.com/code/mariamelseedawy/ts-notebook/output)
2. Download the `saved_models` folder.
3. Rename the `saved_models` folder to `models`.
4. Place the renamed `models` folder in the root directory of the cloned project.

Your project folder structure should look like this:

```
/your-project
    ├── models
    ├── other_project_files
    └── ...
```

### Running the Application

1. **Clone the Repository**:
   ```bash
   git clone <repo-link>
   cd <repo-directory>
   ```

2. **Install Dependencies**:
   If you are not using Docker, install the Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask Server**:
   You can start the Flask development server using:
   ```bash
   python server.py
   ```
   By default, the server will run on `localhost:5000/process`. You can use Postman (import the provided API JSON file) or any API testing tool to interact with the API.

### API Usage

- **Endpoint**: `POST /process`
- **Description**: Make a time series forecast based on the input data.

#### Request Body Example:
```json
{
  "dataset_id": 177,
  "values": [
    {"timestamp": "2021-10-14T00:00:00", "value": 0.263239516},
    {"timestamp": "2021-10-14T00:10:00", "value": 0.226477857},
    {"timestamp": "2021-10-14T00:20:00", "value": 0.249948663},
    {"timestamp": "2021-10-14T00:30:00", "value": 0.250674143},
    {"timestamp": "2021-10-14T00:40:00", "value": 0.254591706},
    {"timestamp": "2021-10-14T00:50:00", "value": 0.256854312},
    {"timestamp": "2021-10-14T01:00:00", "value": 0.224985703},
    {"timestamp": "2021-10-14T01:10:00", "value": 0.281065878}
  ]
}
```

#### Response Body Example:
```json
{
  "prediction": 0.295
}
```

### CSV File
A `min_entries_inference_dataset_with_ids.csv` file is provided that includes:
- **Dataset ID**: A unique identifier for the dataset.
- **Number of Input Values**: The minimum number of time series values required for making a prediction.

The Data Testing Team can use this CSV file to understand the structure and requirements for testing.

### Contact
For any issues or questions, feel free to reach out via Teams or email.
