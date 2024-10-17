import re
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import StringType, FloatType, IntegerType, DoubleType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import functions as F


# 1. Set up the PySpark session
spark = SparkSession.builder.appName("AmazonReviews").getOrCreate()

# 2. Load the dataset 
df = spark.read.csv("gs://cs777_wei_dai/Reviews.csv", header=True, inferSchema=True)

# 3. Text Cleaning Function
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    return text.lower()

# UDF to clean the text
clean_text_udf = udf(lambda x: clean_text(x), StringType())

# Apply the UDF to clean the 'Text' column
df = df.withColumn("CleanedText", clean_text_udf(col("Text")))

# 5. Tokenization
tokenizer = Tokenizer(inputCol="CleanedText", outputCol="words")
df = tokenizer.transform(df)

# 6. Remove Stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
df = remover.transform(df)

# 7. TF-IDF Vectorization
hashing_tf = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=5000)
df = hashing_tf.transform(df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
df = idf.fit(df).transform(df)

# 8.Filter out reviews where HelpfulnessDenominator is 0
df.filter(col("HelpfulnessNumerator").isNull() | col("HelpfulnessDenominator").isNull()).show()
df = df.withColumn("HelpfulnessNumerator", col("HelpfulnessNumerator").cast("int"))
df = df.withColumn("HelpfulnessDenominator", col("HelpfulnessDenominator").cast("int"))
df = df.filter(col("HelpfulnessDenominator") > 0)
df = df.filter(col("HelpfulnessNumerator").isNotNull() & col("HelpfulnessDenominator").isNotNull())

# 9. Create Helpfulness Ratio
df = df.withColumn("HelpfulnessRatio", (col("HelpfulnessNumerator") / col("HelpfulnessDenominator")).cast("float"))

# 10. Create Binary Target (Helpful vs Not Helpful)
df = df.withColumn("HelpfulBinary", (col("HelpfulnessRatio") > 0.5).cast(IntegerType()))
df.printSchema()
df.filter(col("HelpfulBinary").isNull()).show()
df.select("HelpfulnessNumerator", "HelpfulnessDenominator", "HelpfulnessRatio", "HelpfulBinary").show()

# 11. Create Features like Review Length
df = df.withColumn("ReviewLength", size(col("filtered_words")).cast(DoubleType()))

# 12. Convert Score to Numeric (StringIndexer on Score column)
indexer = StringIndexer(inputCol="Score", outputCol="ScoreIndexed", handleInvalid="keep")
df = indexer.fit(df).transform(df)

# Ensure ScoreIndexed and ReviewLength are DoubleType
df = df.withColumn("ScoreIndexed", col("ScoreIndexed").cast(DoubleType()))
df = df.withColumn("ReviewLength", col("ReviewLength").cast(DoubleType()))

# 13. **Create Interaction Features** (Interaction of ScoreIndexed and ReviewLength)
df = df.withColumn("ScoreLengthInteraction", (col("ScoreIndexed") * col("ReviewLength")).cast(DoubleType()))

# 14. Assemble all features into one column (with interaction features)
assembler = VectorAssembler(
    inputCols=["features", "ScoreIndexed", "ReviewLength", "ScoreLengthInteraction"],
    outputCol="finalFeatures",
    handleInvalid="skip"
)
df = assembler.transform(df)

# 15. Scale features
scaler = StandardScaler(inputCol="finalFeatures", outputCol="scaledFeatures")
df = scaler.fit(df).transform(df)

# 16. Select final features and label for model training
final_df = df.select("scaledFeatures", "HelpfulBinary")
final_df = final_df.withColumnRenamed("HelpfulBinary", "label")

# Ensure the label is of type Double
final_df = final_df.withColumn("label", col("label").cast(DoubleType()))

# 16. Split the data into training and testing sets
train_data, test_data = final_df.randomSplit([0.8, 0.2], seed=42)

# 17. Calculate the imbalance ratio
class_counts = train_data.groupBy('label').count().collect()
minority_class_count = min(class_counts, key=lambda x: x['count'])['count']
majority_class_count = max(class_counts, key=lambda x: x['count'])['count']
imbalance_ratio = majority_class_count / minority_class_count

# Adjust class weights based on imbalance ratio
train_data = train_data.withColumn("classWeights", F.when(col("label") == 1, imbalance_ratio).otherwise(1.0))

# 18. Define models with class weights
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label', weightCol="classWeights")
rf = RandomForestClassifier(featuresCol='scaledFeatures', labelCol='label')
svm = LinearSVC(featuresCol='scaledFeatures', labelCol='label', weightCol="classWeights")

# 19. Define hyperparameter grids
param_grid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

param_grid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 50]) \
    .addGrid(rf.maxDepth, [5, 10, 20]) \
    .addGrid(rf.minInstancesPerNode, [1, 5]) \
    .build()

param_grid_svm = ParamGridBuilder() \
    .addGrid(svm.maxIter, [50, 100, 200]) \
    .addGrid(svm.regParam, [0.01, 0.1, 1.0]) \
    .build()


# 20. Cross-validator setup
crossval_lr = CrossValidator(estimator=lr,
                             estimatorParamMaps=param_grid_lr,
                             evaluator=MulticlassClassificationEvaluator(),
                             numFolds=5)

crossval_rf = CrossValidator(estimator=rf,
                             estimatorParamMaps=param_grid_rf,
                             evaluator=MulticlassClassificationEvaluator(),
                             numFolds=5)

crossval_svm = CrossValidator(estimator=svm,
                              estimatorParamMaps=param_grid_svm,
                              evaluator=MulticlassClassificationEvaluator(),
                              numFolds=5)

# 21. Train models
cv_model_lr = crossval_lr.fit(train_data)
cv_model_rf = crossval_rf.fit(train_data)
cv_model_svm = crossval_svm.fit(train_data)

# 22. Get predictions
predictions_lr_cv = cv_model_lr.transform(test_data)
predictions_rf_cv = cv_model_rf.transform(test_data)
predictions_svm_cv = cv_model_svm.transform(test_data)

# 23. Evaluation function
def evaluate_model(predictions, model_name):
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    
    evaluator_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
    precision = evaluator_precision.evaluate(predictions)
    
    evaluator_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
    recall = evaluator_recall.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator_f1.evaluate(predictions)
    
    print(f"--- {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print("-" * 30)

# 24. Confusion Matrix function
def confusion_matrix(predictions, model_name):
    prediction_and_labels = predictions.select("prediction", "label").rdd
    metrics = MulticlassMetrics(prediction_and_labels)
    confusion = metrics.confusionMatrix().toArray()
    print(f"--- Confusion Matrix for {model_name} ---")
    print(confusion)
    print("-" * 30)

# Evaluate models
evaluate_model(predictions_lr_cv, "Logistic Regression with Cross-Validation")
confusion_matrix(predictions_lr_cv, "Logistic Regression with Cross-Validation")

evaluate_model(predictions_rf_cv, "Random Forest with Cross-Validation")
confusion_matrix(predictions_rf_cv, "Random Forest with Cross-Validation")

evaluate_model(predictions_svm_cv, "SVM with Cross-Validation")
confusion_matrix(predictions_svm_cv, "SVM with Cross-Validation")

# 25. Print best parameters from the CrossValidator setup
def print_best_params_from_cv(crossval_model, param_grid, model_name):
    print(f"--- Best Parameters for {model_name} ---")

    best_params = crossval_model.getEstimatorParamMaps()[np.argmax(crossval_model.avgMetrics)]

    for param, value in best_params.items():
        print(f"{param.name}: {value}")

    print("-" * 30)

# Print best parameters for each model using the CrossValidator
print_best_params_from_cv(cv_model_lr, param_grid_lr, "Logistic Regression")
print_best_params_from_cv(cv_model_rf, param_grid_rf, "Random Forest")
print_best_params_from_cv(cv_model_svm, param_grid_svm, "SVM")

# Get the best model from the CrossValidator for Random Forest
best_rf_model = cv_model_rf.bestModel

# Access the feature importances
importances = best_rf_model.featureImportances

# List of feature names used in the model
feature_names = ["TF-IDF Features", "ScoreIndexed", "ReviewLength", "ScoreLengthInteraction"]

# Display the feature importances for each feature
for idx, feature_name in enumerate(feature_names):
    print(f"{feature_name}: {importances[idx]}")

# Get the best model from the CrossValidator for Logistic Regression
best_lr_model = cv_model_lr.bestModel

# Access the coefficients
coefficients = best_lr_model.coefficients

# List of feature names used in the model
feature_names = ["TF-IDF Features", "ScoreIndexed", "ReviewLength", "ScoreLengthInteraction"]

# Display the coefficients for each feature
for idx, feature_name in enumerate(feature_names):
    print(f"{feature_name}: {coefficients[idx]}")

# Get the best model from the CrossValidator for SVM
best_svm_model = cv_model_svm.bestModel

# Access the coefficients
coefficients = best_svm_model.coefficients

# List of feature names used in the model
feature_names = ["TF-IDF Features", "ScoreIndexed", "ReviewLength", "ScoreLengthInteraction"]

# Display the coefficients for each feature
for idx, feature_name in enumerate(feature_names):
    print(f"{feature_name}: {coefficients[idx]}")
