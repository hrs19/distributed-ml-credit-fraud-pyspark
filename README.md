# Machine Learning Algorithms with PySpark

This repository contains multiple machine learning algorithms implemented using PySpark. The scripts are organized into different folders, each focusing on a specific machine learning model. The aim is to leverage PySpark's capabilities for both serial and parallel processing to handle large datasets efficiently.

## Overview

This project includes implementations of various popular machine learning models using PySpark. These implementations cover both serial and parallel processing approaches to demonstrate how different models can benefit from distributed data processing.

## Folder Structure

```
.
├── KNN/
│   ├── knn_par.py
│   └── knn_serial.py
├── LogisticRegression/
│   ├── LR.py
│   ├── LogReg.py
│   ├── README.md
│   ├── SerialLR.py
│   └── weightedLR.py
├── RandomForest/
│   ├── parallel_random_forest.py
│   ├── random_forest_parameters.txt
│   └── serial_random_forest.py
└── SVM/
    ├── SVM_Parallel.py
    └── SVM_Serial.py
```

## Files Description

- **KNN/**: Contains implementations of the K-Nearest Neighbors algorithm.
  - `knn_par.py`: Parallel implementation of KNN using PySpark.
  - `knn_serial.py`: Serial implementation of KNN.

- **LogisticRegression/**: Contains implementations of Logistic Regression.
  - `SerialLR.py`: Serial implementation of Logistic Regression.
  - `weightedLR.py`: Implementation of weighted Logistic Regression.
  - `LR.py` and `LogReg.py`: Additional logistic regression implementations with optimizations for parallel processing.

- **RandomForest/**: Contains implementations of the Random Forest algorithm.
  - `serial_random_forest.py`: Serial implementation of Random Forest.
  - `parallel_random_forest.py`: Parallel implementation of Random Forest using PySpark.
  - `random_forest_parameters.txt`: Configuration parameters for Random Forest experiments.

- **SVM/**: Contains implementations of Support Vector Machine (SVM).
  - `SVM_Serial.py`: Serial implementation of SVM.
  - `SVM_Parallel.py`: Parallel implementation of SVM using PySpark.

## How to Run on the Discovery Cluster

### Submitting Jobs

To run the Logistic Regression scripts on the Discovery cluster at Northeastern University, follow these steps:

1. **Access the Discovery Cluster**

   You need access to the Discovery cluster. You can connect via SSH:
   
   ```sh
   ssh <your_username>@login.discovery.neu.edu
   ```

2. **Load Required Modules**

   Load the necessary modules to ensure that PySpark and the required versions of Python are available:

   ```sh
   module load python/3.x
   module load spark/3.x
   ```

3. **Upload Your Scripts**

   Upload the script you wish to run to the Discovery cluster using `scp`:

   ```sh
   scp SerialLR.py <your_username>@xfer.discovery.neu.edu:/work/<your_username>/
   ```

4. **Create a Job Submission Script**

   Create a job submission script (`spark_job.sh`) for your PySpark job:

   ```sh
   #!/bin/bash
   #SBATCH --job-name=pyspark_job        # Job name
   #SBATCH --nodes=1                     # Number of nodes
   #SBATCH --ntasks-per-node=16          # Number of tasks per node (adjust as needed)
   #SBATCH --time=01:00:00               # Runtime (HH:MM:SS)
   #SBATCH --output=output_%j.log        # Output file (%j will be replaced by job ID)

   module load python/3.x
   module load spark/3.x

   spark-submit SerialLR.py
   ```

   Adjust the resource requirements (`--nodes`, `--ntasks-per-node`, etc.) based on the requirements of your specific job and the limits of the cluster.

5. **Submit the Job to the Scheduler**

   Submit your job script to the Discovery cluster using `sbatch`:

   ```sh
   sbatch spark_job.sh
   ```


## Acknowledgements

- This implementation uses the PySpark MLlib library, which provides an efficient API for machine learning with Apache Spark.
- PySpark documentation: [https://spark.apache.org/docs/latest/api/python/](https://spark.apache.org/docs/latest/api/python/)
- Discovery Cluster documentation: https://rc.northeastern.edu/documentation/
