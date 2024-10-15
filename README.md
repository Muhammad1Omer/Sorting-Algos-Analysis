# Sorting Algorithms Visualization Platform

An interactive platform built with live animations to visually demonstrate and compare the performance of various sorting algorithms. The project showcases full-stack development, algorithm performance analysis, and data visualization.

## Features

- **Live Sorting Algorithm Animations**: Visual demonstration of how different sorting algorithms work.
- **Random Dataset Generation**: The `generate_dataset` function generates datasets of varying sizes for testing sorting algorithms.
- **Time Complexity Measurement**: Execution time of each algorithm is measured using the `timeit` module, and results are stored in a MongoDB collection for further analysis.
- **Data Visualization**: The execution time results are stored in **Postgres** and visualized using **Apache Superset** to create insightful charts.

## Technologies Used

- **Django**: Manages frontend-to-backend communication.
- **MongoDB**: Stores and manages sorting algorithm performance results.
- **Postgres**: Handles data conversion and storage for visualization.
- **Apache Superset** (Dockerized): Used to generate charts and dashboards for data visualization.
- **Streamlit**: Provides an interactive web app interface to showcase live sorting algorithm animations.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>

# Project Setup

## Install dependencies

Navigate to the project directory and install the necessary dependencies:

  ```bash
  pip install Django pymongo psycopg2 streamlit time numba
  ```

## Set up MongoDB and Postgres

- **MongoDB**: Install and configure MongoDB for storing performance results.
- **Postgres**: Install and configure Postgres for saving processed data.

Update your database configurations in the `Django settings.py` file.


## Run the Django server

Launch the Django server:

```bash
python manage.py runserver
```

## Run the Streamlit app

To visualize the live sorting algorithm animations:

```bash
streamlit run app.py
```


## Usage

Open the Streamlit interface in your browser to interact with the sorting algorithm animations. The `timeit` module will measure the sorting algorithms' performance, and the results will be stored in MongoDB. Use Apache Superset (running in Docker) to visualize the data stored in Postgres with customizable charts and dashboards.

## Contributors

- **Muhammad Omer**
- Azeem Chaudhary
- Muhammad Shariq Usman

## License

This project is licensed under the MIT License - see the LICENSE file.




