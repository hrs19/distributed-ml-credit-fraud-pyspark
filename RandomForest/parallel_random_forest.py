#import findspark
#findspark.init("/shared/centos7/spark/2.4.5-hadoop2.7")
import pyspark
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.feature import StandardScaler, StandardScalerModel
from pyspark.mllib.evaluation import RankingMetrics
from time import time

def min_array(x,y):
    z = []
    for i in range(len(x)):
        z.append(min(x[i],y[i]))
    
    return np.array(z)

def max_array(x,y):
    z = []
    for i in range(len(x)):
        z.append(max(x[i],y[i]))
    
    return np.array(z)

def standardScaling(rdd):
    standardizer = StandardScaler(True,True)
    scaled_data_vals = standardizer.fit(rdd.values())
    mean_array = np.array(scaled_data_vals.mean)
    std_array = np.array(scaled_data_vals.std)
    scaled_data = rdd.mapValues(lambda x: (x-mean_array)/std_array)
    
    return scaled_data

def min_maxScaling(rdd):
    minimum = rdd.values().reduce(min_array)
    maximum = rdd.values().reduce(max_array)
    scaled_data = rdd.mapValues(lambda x: (x-minimum)/(maximum-minimum))

    return scaled_data

sc = SparkContext("local[50]",appName="Parallel Random Forests")
sc.setLogLevel("warn")

start_time = time()
raw_data_rdd = sc.textFile('/scratch/tarasia.dev/Project/creditcard.csv',minPartitions=100).map(lambda x: x.split(',')).map(lambda x: (x[-1],x[:-1]))\
                .map(lambda x: (float(x[0][1]),np.array([float(y) for y in x[1]])))#.map(lambda x: LabeledPoint(label=x[0],features=x[1]))

end_time = time()
print("Time Taken to read actual data: {:1.5f}".format(end_time-start_time))


start_time = time()
standard_scale = standardScaling(raw_data_rdd)
end_time = time()
print("Time Taken to scale actual data: {:1.5f}".format(end_time-start_time))

min_max_scale = min_maxScaling(raw_data_rdd)

standard_scale = standard_scale.map(lambda x: LabeledPoint(label=x[0],features=x[1]))
min_max_scale = min_max_scale.map(lambda x: LabeledPoint(label=x[0],features=x[1]))

test, train = standard_scale.randomSplit(weights=[0.3, 0.7], seed=1)
start_time = time()
model = RandomForest.trainClassifier(train, numClasses=2, categoricalFeaturesInfo={}, numTrees=100, featureSubsetStrategy="auto", impurity="gini", maxDepth=10, maxBins=32)
end_time = time()
print("Time Taken to fit oversampled data: {:1.5f}".format(end_time-start_time))

start_time = time()
model = SVMWithSGD.train(train, regType="l2")
end_time = time()
print("Time Taken to fit actual data: {:1.5f}".format(end_time-start_time))

start_time = time()
predictions = model.predict(test.map(lambda x: x.features))
end_time = time()
print("Time Taken to test actual data: {:1.5f}".format(end_time-start_time))
#predictions_gndTruth = predictions.zip(test.map(lambda x: x.label))
#print(predictions_gndTruth.take(1))
#metrics = RankingMetrics(predictions_gndTruth)

#print(end_time-start_timei)
#print(metrics.meanAveragePrecision)
