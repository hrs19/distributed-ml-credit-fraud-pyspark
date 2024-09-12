from pyspark import SparkContext
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from time import time

# Create SparkContext
sc = SparkContext(appName="CreditCardFraud")
sc.setLogLevel('warn')

read_data_time = time()
# Load data 
credit_card_data_rdd = sc.textFile("oversample.csv").map(lambda line: line.split(","))

# credit_card_data_rdd = sc.textFile("creditcard1.csv").map(lambda line: line.split(","))
read_data_time = time()-read_data_time

header = credit_card_data_rdd.first()

credit_card_data_rdd = credit_card_data_rdd.filter(lambda row: row != header)

(training_data_rdd, test_data_rdd) = credit_card_data_rdd.randomSplit([0.7, 0.3], seed=1234)

# Convert the data to LabeledPoints
training_data_rdd = training_data_rdd.map(lambda row: LabeledPoint(float(row[-1]), Vectors.dense([float(x) for x in row[:-1]])))
test_data_rdd = test_data_rdd.map(lambda row: LabeledPoint(float(row[-1]), Vectors.dense([float(x) for x in row[:-1]])))
print('data:',training_data_rdd.take(2))

def scale_data(data_rdd, scaler_mean, scaler_std):
    scaled_rdd = data_rdd.map(lambda row: LabeledPoint(row.label, (row.features - scaler_mean) / scaler_std))
    return scaled_rdd

scaler = StandardScaler(withMean=True, withStd=True).fit(training_data_rdd.map(lambda row: row.features))

scaler_mean = sc.broadcast(scaler.mean)
scaler_std = sc.broadcast(scaler.std)

scale_train_time = time()
training_data_rdd = scale_data(training_data_rdd, scaler_mean.value, scaler_std.value)
scale_train_time = time()-scale_train_time

scale_test_time = time()

test_data_rdd = scale_data(test_data_rdd, scaler_mean.value, scaler_std.value)
scale_test_time = time()-scale_test_time

fit_time = time()
log_reg_model = LogisticRegressionWithLBFGS.train(training_data_rdd, iterations=1000, numClasses=2, regParam=0.01)
fit_time = time()-fit_time

predictions_and_labels_rdd = test_data_rdd.map(lambda row: (float(log_reg_model.predict(row.features)), row.label))


multiclass_metrics = MulticlassMetrics(predictions_and_labels_rdd)

precision = multiclass_metrics.precision()
recall = multiclass_metrics.recall()
f1_score = multiclass_metrics.fMeasure()

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1_score))

print(f'Time taken: \nReadDataRDD: {read_data_time}\tScale Train set: {scale_train_time}\tScale Test set: {scale_test_time}\tLR fit time: {fit_time}')
sc.stop()
