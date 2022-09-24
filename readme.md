#### geometry_2d.py:
It contains the solutions for the question 1 corresponding the 2d geometry questions.
In or the to test the provided solution, you can run **test_geometry_2d.py**.

#### geometry_3d.py:
It contains the solutions for the question 2 corresponding the 2d geometry questions.
In or the to test the provided solution, you can run **test_geometry_3d.py**.

#### custom_exceptions.py:
It provides a few custom exceptions class that derived from Exception class. 
These custom exceptions is used in the solutions provided for the **Question 1 and 2**.

### visualization_ML.py:
This file contains solution for the question 3 corresponding to the visualization and machine learning questions.

#### How to run the visualization_ML.py

**Generate samples (Q1)**
```
python visualization_ML.py --generate_data --file_path ./data.npy
```

**Generate samples and visualize the generated data (Q1 and Q2)**
```
python visualization_ML.py --generate_data --file_path ./data.npy --plot_samples
```

** Only visualize the provided test data (Q2)**
```
python visualization_ML.py --file_path ./sample_data.npy --plot_samples
```

** generate, visualize the data, and plot the hypermarket tuning figure**
```
python visualization_ML.py --generate_data --file_path ./data.npy --plot_samples --ht_tuning
```

** generate, visualize the data, plot the hypermarket tuning, and plt the clustering result **
```
python visualization_ML.py --generate_data --file_path ./data.npy --plot_samples --ht_tuning --plot_clusters --dbscan_min_sample 3 --dbscan_epsilon 0.49 --seed 10 
```

** generate, visualize the data, plot the hypermarket tuning, and plt the clustering result for the provided test data file **
```
python visualization_ML.py --file_path ./sample_data.npy --plot_samples --ht_tuning --plot_clusters --dbscan_min_sample 3 --dbscan_epsilon 2.5 --seed 10 
```

