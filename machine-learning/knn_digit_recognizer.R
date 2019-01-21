"Loading the Required Libraries"
library(caret)

"setwd"
setwd("/Users/VRC/Documents/ISU-Fall/cs617/lab04_ann_project/")

"Read the training data"
training_data = read.csv("train.csv")

"Basic Summary of the training data"
dim(training_data)
str(training_data)
class(training_data$label)

"Converting the Label to Factor (The Actual Digits Column)"
training_data$label = as.factor(training_data$label)
"Remove the Label Column(First One) from the traning data 
To make sure we have a dataframe with only data."
training_data_wt_label=training_data[,-1]

"New Dataset dimensions - Without Label Column"
dim(training_data_wt_label)

"Eliminating the Columns with Near Zero Variance"
near_zero_var_cols = nearZeroVar(training_data_wt_label,saveMetrics=TRUE)

"Colums Removed due to Near Zero Variance"
which(near_zero_var_cols$nzv==TRUE)

"Eliminating the columns that are predicted to be Near zero Variance"
training_data_wt_label = training_data_wt_label[,near_zero_var_cols$nzv==FALSE]

"Dimensions after Reduction"
dim(training_data_wt_label)

"Data optimization using Centering & PCA"
pca_object = preProcess(training_data_wt_label,method = c("center", "pca"))

"Pre-compute data using the newly predicted PCA Vectors"
pre_compute = predict(pca_object,training_data_wt_label)

"Dimensions of newly computed object"
dim(pre_compute)
class(pre_compute)

"Training the data using the Cross Validation method with 10 rounds"
ctrl_object = trainControl(method = "cv", # Using cross-validation
                    number = 10) # Use 10 folds or cv

"Training the model"
model_knn = train(pre_compute,training_data$label, 
                  method = "knn",
                  trControl = ctrl_object)

"Summary of the Model"
model_knn
summary(model_knn)

"Read the test data"
test_data = read.csv("test.csv")

"Basic Summary of the training data"
dim(test_data)
str(test_data)

"Eliminating the Columns with Near Zero Variance"
test_data = test_data[,near_zero_var_cols$nzv==FALSE]

"Colums Removed due to Near Zero Variance"
dim(test_data)

"Pre-compute data using predicted PCA Vectors"
test_data = predict(pca_object,test_data)
dim(test_data)

"Predicting for the test data"
test_data$label = predict(model_knn, newdata = test_data)
test_data$ImageId = 1:nrow(test_data)

"Creating Output vectors"
output = test_data[,c("ImageId","label")]

"Writing the results to a file"
write.table(output, file = "output.csv", col.names = TRUE, row.names = FALSE, sep = ",")


