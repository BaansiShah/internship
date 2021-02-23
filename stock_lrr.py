# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 12:19:03 2021

@author: bansi
"""

from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import matplotlib.pyplot as plt
import datetime
earlier = datetime.datetime.now()
sc = SparkContext('local')
spark = SparkSession(sc)


dataset = spark.read.csv("C:/Users/bansi/spark-3.0.1-bin-hadoop2.7/Stock Data Google.csv",inferSchema=True,header=True)


featureassembler=VectorAssembler(inputCols=["Open","High","Low","Volume"],outputCol="Independent Features")

output=featureassembler.transform(dataset)

finalized_data=output.select("Independent Features","Close")
train_data,test_data=finalized_data.randomSplit([0.75,0.25])
regressor=LinearRegression(featuresCol='Independent Features', labelCol='Close')
regressor=regressor.fit(train_data)
predictions=regressor.transform(test_data)
#predictions.show()


lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Close",metricName="r2")
test_result = regressor.evaluate(test_data)

print("R Squared (R2) = %g" % lr_evaluator.evaluate(predictions))
print("Root Mean Squared Error (RMSE) = %g" % test_result.rootMeanSquaredError)
print("Mean Absolute Error = %g" % test_result.meanAbsoluteError)
print("Mean Squared Error = %g" % test_result.meanSquaredError)


actual = test_data.toPandas()['Close'].values.tolist()
predicted = predictions.toPandas()['prediction'].values.tolist()

plt.figure(figsize=(20,10))
plt.plot(actual, label='Actual', color='green')
plt.plot(predicted, color='red', label='Predicted')
plt.legend(loc="upper left")
now = datetime.datetime.now()
diff = now - earlier

print(" Delay in seconds = %d " %diff.seconds)
plt.show()



