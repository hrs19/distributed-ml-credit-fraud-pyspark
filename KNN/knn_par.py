from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import KNNClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("project/creditcard.csv")

training_data, validation_data, test_data = data.randomSplit([0.7, 0.15, 0.15])

feature_cols = data.columns[:-1]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
training_data = assembler.transform(training_data)
validation_data = assembler.transform(validation_data)
test_data = assembler.transform(test_data)

knn = KNNClassifier(k=5, distanceMeasure="euclidean", lambda_=450, mu=450)

start_time = time.time()
model = knn.fit(training_data)
end_time = time.time()
train_time = end_time - start_time

start_time = time.time()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
accuracy = evaluator.evaluate(model.transform(validation_data))
end_time = time.time()
val_time = end_time - start_time

print(f"Validation accuracy: {accuracy}")
print(f"Training time: {train_time:.2f} seconds")
print(f"Validation time: {val_time:.2f} seconds")

start_time = time.time()
accuracy = evaluator.evaluate(model.transform(test_data))
end_time = time.time()
test_time = end_time - start_time

print(f"Test accuracy: {accuracy}")
print(f"Test time: {test_time:.2f} seconds")
