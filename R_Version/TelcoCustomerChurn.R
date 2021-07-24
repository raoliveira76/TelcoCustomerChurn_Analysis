# --------------------------------------------------------------------
# PROJECT TELCO CUSTOMER CHURN
# Predicting Behaviour to Retain Customers using Classification Models
#
# Krystine L. Yap (ID 300307908)
# Mateus Aldi Christy (ID 300312651)
# Rafael S. Menezes Pires (ID 300301161)
# Ronaldo A. de Oliveira (ID 300305192)
#
# November 25, 2020
#
# --------------------------------------------------------------------


# --------------
# LOAD LIBRARIES
# --------------
library(caret)
library(dplyr)
library(ggplot2)
library(ISLR)
library(fastDummies)


# ----------------
# GENERAL SETTINGS
# ----------------
setwd("D:/R/Prj_CSIS3360/")
set.seed(2020)


# ------------------------------
# LOAD AND PREPROCESSING DATASET
# ------------------------------
# read dataset
telcoDataset <- read.csv("TelcoCustomerChurn.csv")

# delete "CustomerID" column
telcoDataset <- telcoDataset[2:21]

# delete all rows with a null value
telcoDataset <- na.omit(telcoDataset)

# check if all rows with null were deleted
telcoDataset[!complete.cases(telcoDataset), ]

# show the first 6 rows
head(telcoDataset)

# summary of attributes
summary(telcoDataset)
str(telcoDataset)

# types of attributes
sapply(telcoDataset, class)

# Outcome distribution
perc <- prop.table(table(telcoDataset$Churn)) * 100
cbind(freq=table(telcoDataset$Churn), percentage=perc)


# --------------------
# EXPLORATORY ANALYSIS
# --------------------
options(repr.plot.width = 15, repr.plot.height = 10)

# Churn proportion
telcoDataset %>% 
    group_by(Churn) %>% 
    summarise(Count = n()) %>% 
    mutate(percent = prop.table(Count) * 100) %>%
    ggplot(aes(reorder(Churn, percent), percent), fill=Churn) +
    geom_col(fill = c("#5577FF", "#55FF77"), alpha=0.5)+
    geom_text(aes(label = sprintf("%.1f%%", percent)), vjust=-0.8, size=4) +
    theme_bw()+
    theme(legend.position = "right") +
    xlab("Churn") + 
    ylab("Percent")+
    ggtitle("Churn Proportion")

# Churn by Gender
p <- ggplot(telcoDataset, aes(x=Churn, fill=gender)) + geom_bar(stat='count', alpha=0.35, position="dodge")
p + scale_color_brewer(palette="Accent") + 
    labs(title="Quantity of Churn Types by Gender", x="Churn", y = "") +
    theme(legend.position = "right") +
    theme(plot.title = element_text(hjust = 0.5))

# Churn by Dependents
p <- ggplot(telcoDataset, aes(x=Churn, fill=Dependents)) + geom_bar(stat='count', alpha=0.35, position="dodge")
p + scale_color_brewer(palette="Accent") + 
    labs(title="Quantity of Churn Types by Gender", x="Churn", y = "") +
    theme(legend.position = "right") +
    theme(plot.title = element_text(hjust = 0.5)) 

# Churn by InternetService
p <- ggplot(telcoDataset, aes(x=Churn, fill=InternetService)) + geom_bar(stat='count', alpha=0.35, position="dodge")
p + scale_color_brewer(palette="Accent") + 
    labs(title="Quantity of Churn Types by Gender", x="Churn", y = "") +
    theme(legend.position = "right") +
    theme(plot.title = element_text(hjust = 0.5)) 

# Churn by Contract
p <- ggplot(telcoDataset, aes(x=Churn, fill=Contract)) + geom_bar(stat='count', alpha=0.35, position="dodge")
p + scale_color_brewer(palette="Accent") + 
    labs(title="Quantity of Churn Types by Contract", x="Churn", y = "") +
    theme(legend.position = "right") +
    theme(plot.title = element_text(hjust = 0.5)) + 
    ylim(0, 2500)


# ----------------------------
# DATA PROCESSING AND MODELING
# ----------------------------
# run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# create the training and testing data sets (70% / 30%)
testing_index <- createDataPartition(telcoDataset$Churn, p=0.70, list=FALSE)
testing <- telcoDataset[-testing_index, ]
training <- telcoDataset[testing_index, ]

# show the dimensions of each data sets
dim(training)
dim(testing)

# Classification and Regression Tree (CART)
fit.cart <- train(Churn~., data=training, method="rpart", metric=metric, trControl=control)

# k-Nearest Neighbors (kNN)
fit.knn <- train(Churn~., data=training, method="knn", metric=metric, trControl=control)

# Support Vector Machine
fit.svm <- train(Churn~., data=training, method="svmRadial", metric=metric, trControl=control)

# Random Forest
fit.rf <- train(Churn~., data=training, method="rf", metric=metric, trControl=control)

# summarize accuracy of models
results <- resamples(list(cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

# compare accuracy of models
dotplot(results)

# summary of the best model
print(fit.svm)

# precision analysis
predictions <- predict(fit.svm, testing)
confusionMatrix(predictions, as.factor(testing$Churn))

# determining the attributes that are significant statistically using logistic regression and creating dummy variables
telco_dum <- dummy_cols(telcoDataset, 
                        select_columns = c('gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                                           'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                           'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                                           'PaperlessBilling', 'PaymentMethod', 'Churn'), 
                        remove_first_dummy = TRUE, 
                        remove_selected_columns = TRUE)

# model with all variables
logreg_all <- glm(Churn_Yes ~ ., 
               data= telco_dum, 
               family = 'binomial')
summary(logreg_all)

# model with only significant statistically variable
logreg_sig <- glm(Churn_Yes ~ SeniorCitizen + tenure + TotalCharges + MultipleLines_Yes + 
                   `InternetService_Fiber optic` + InternetService_No + `Contract_One year` + 
                   `Contract_Two year` + PaperlessBilling_Yes + `PaymentMethod_Electronic check` , 
               data= telco_dum, 
               family = 'binomial')
summary(logreg_sig)

# compute the odds ratio
exp(coef(logreg_sig))


# -----------
# PREDICTIONS
# -----------
# customer #1 => Female with partner but no dependent; 15 years tenure, monthly contract, no streaming TV, 
#			device protection, and online security, but with online backup, tech support and multiple lines
#       	monthly charges $50 and $250 in total paid by electronic check.
customer1 <- data.frame('gender' = 'Female', "SeniorCitizen" = 0, "Partner" = 'Yes', "Dependents" = 'No',
                        'tenure' = 15, 'PhoneService' = 'Yes', 'MultipleLines' = 'Yes', 
                        'InternetService' = 'Fiber optic', 'OnlineSecurity' = 'No','OnlineBackup' = 'Yes', 
                        'DeviceProtection' = 'No','TechSupport' = 'Yes', 'StreamingTV' = 'No', 
                        'StreamingMovies' = 'No','Contract' = 'Month-to-month', 'PaperlessBilling' = 'No',
                        'PaymentMethod' = 'Electronic check', 'MonthlyCharges' = 50, 'TotalCharges' = 250)
predict(fit.svm, customer1)

# customer #2 => Male, no dependents or partner; 3 years tenure, monthly contract, with streaming TV and 
#			online security, but no device protection, tech support, and streaming movies;
#	       	monthly charges $100 and $300 in total paid by automatic bank transfer.
customer2 <- data.frame('gender' = 'Male', "SeniorCitizen" = 0, "Partner" = 'No', "Dependents" = 'No',
                        'tenure' = 3, 'PhoneService' = 'Yes', 'MultipleLines' = 'No', 
                        'InternetService' = 'Fiber optic', 'OnlineSecurity' = 'No','OnlineBackup' = 'Yes', 
                        'DeviceProtection' = 'No','TechSupport' = 'No', 'StreamingTV' = 'Yes', 
                        'StreamingMovies' = 'No','Contract' = 'Month-to-month', 'PaperlessBilling' = 'No',
                        'PaymentMethod' = 'Bank transfer (automatic)', 'MonthlyCharges' = 100, 'TotalCharges' = 300)
predict(fit.svm, customer2)
