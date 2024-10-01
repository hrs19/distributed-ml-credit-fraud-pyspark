# Logistic Regression with PySpark for Parallel Processing

This repository contains Python scripts that implement Logistic Regression using PySpark for parallel processing. Logistic Regression is a widely used statistical model for binary classification, and in this project, we leverage PySpark to handle large datasets efficiently by utilizing parallel processing.


## Overview

The scripts provided demonstrate how to perform Logistic Regression on datasets using PySpark, which allows for efficient processing of large data through parallel computing. This implementation is ideal for distributed systems and provides a practical demonstration of how to utilize PySpark's MLlib for machine learning tasks.

## Files in this Repository

- `SerialLR.py`: Implements a simple Logistic Regression model without parallel optimizations, demonstrating the basic setup of a logistic regression using PySpark.
- `weightedLR.py`: Implements a weighted Logistic Regression, where different data points can have different weights during the training process.
- `LogReg.py` and `LR.py`: Additional Logistic Regression implementations with different variations or optimizations for parallel processing using PySpark.

## Usage

### Serial Logistic Regression

The `SerialLR.py` script provides a simple implementation of Logistic Regression using PySpark without any advanced weighting or parallel considerations. This script is useful for understanding the basic components of logistic regression, including data loading, feature extraction, and model fitting.

### Weighted Logistic Regression

The `weightedLR.py` script adds weighting capabilities to Logistic Regression. This allows for giving different importance to certain data points, which can be useful in scenarios like dealing with imbalanced datasets.


## How to Run on the Discovery Cluster

### Submitting Jobs

To run the Logistic Regression scripts on the Discovery cluster at Northeastern University, follow these steps:

1. **Access the Discovery Cluster**

   You need access to the Discovery cluster. You can connect via SSH:
   
   ```sh
   ssh <your_username>@discovery.rc.northeastern.edu
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
   scp SerialLR.py <your_username>@discovery.rc.northeastern.edu:/work/<your_username>/
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
