# Load Libraries
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)

# We also want to make sure we don't set the seed since we want different results each time.
# Read in csv of just ovarian cancer
#  We create a randomized matrix of numbers and go through the same 
df <- as.data.frame(matrix(rnorm(n=374*48), nrow=374, ncol=48))
labels <- strsplit("XRCC6 XRCC5 XRCC7 LIG4 LIG3 LIG1 XRCC4 NHEJ1 XRCC1 DCLRE1C TP53BP1 BRCA1 BRCA2 EXO1 EXD2 POLM POLL POLQ RAD50 MRE11 NBN TDP1 RBBP8 CTBP1 APLF PARP1 PARP3 PNKP APTX WRN PAXX RIF1 RAD52 RAD51 ATM ATR TP53 H2AX ERCC1 ERCC4 RPA1 MSH2 MSH3 RAD1 MSH6 PMS1 MLH1 MLH3", " +")[[1]]
colnames(df) <- labels

# Create 
df$SURVIVAL = as.data.frame(matrix(sample(c("TRUE", "FALSE"), 374, replace=TRUE, prob=c(0.387, 0.612)), nrow=374))

# Shuffle the rows of the data-set to remove patterns
random_order = sample(nrow(df))
df = df[random_order,]


# T-Test
calculate = function(x1, x2){
  for(col in colnames(df)){
    # Skip non numeric columns
    if(col == "SURVIVAL"){
      next
    }
    # Compute Z-Score for normalization
    u1 = mean(x1[,col])
    u2 = mean(x2[,col])
    sigma1 = sd(x1[,col])
    sigma2 = sd(x2[,col])
    n1 = length(x1[,col])
    n2 = length(x2[,col])
    t = (u1 - u2) / sqrt((sigma1^2 / n1) + (sigma2^2 / n2))
    
    p = pt(t, n1 + n2 - 2)
    pos_cor = FALSE
    if (t > 0){
      p = 1 - p
      pos_cor = TRUE
    }
    print(paste(col, t, p, pos_cor))
  }
}

s1 = df[df$SURVIVAL == "TRUE",]
s2 = df[df$SURVIVAL == "FALSE",]

print("SURVIVAL T-value P-value Gene-Favors-Survival")
calculate(s1, s2)

# Normalize the columns
for(col in colnames(df)){
  # Skip non numeric columns
  if(col == "SURVIVAL"){
    next
  }
  # Compute Z-Score for normalization
  avg = mean(df[,col])
  std = sd(df[,col])
  df[,col] = (df[,col] - avg) / std
}

# Making data-sets
# Fix original df Convert survival to bool
df$SURVIVAL = as.integer(df$SURVIVAL == "TRUE")

# Make a function for creating cross-validation numbers
cross_val <- function(n, k=10){
  size_set = n / k
  indexes = vector(mode="list", length=k)
  for (i in seq(0, k-1)){
    set_nums = list(seq(floor(i*size_set)+1, floor((i+1)*size_set)))
    indexes[i+1] = set_nums
  }
  return(indexes)
}

# Make an array of row numbers to make training and testing sets for 10-fold cross-validation
survival_index = cross_val(nrow(df))

# Function for accuracy calculations
evalr <- function(model, test, actual, type){
  preds = predict(model, test, type=type)
  if (type == "response"){
    preds = preds > 0.5
  } else if (type == "class") {
    preds = preds == 1
  }
  lister = preds == actual
  acc = sum(lister) / length(lister)
  mtrx = confusionMatrix(factor(preds, levels=c(TRUE, FALSE)), factor(actual == 1, levels=c(TRUE, FALSE)), mode = "everything")$byClass
  return (c(acc, mtrx[['Balanced Accuracy']]))
}


# Create logistic regression model for the different predictors 
# Iterate through validation sets
sum_surv_acc = 0
sum_surv_f = 0
for (t_set in survival_index){
  train = df[-t_set,]
  test = df[t_set,]
  
  log_model_survival = glm(SURVIVAL~TP53BP1+CTBP1+APTX+RAD52+PMS1, data=train)
  pair_acc_f = evalr(log_model_survival, test, test$SURVIVAL, "response")
  sum_surv_acc = sum_surv_acc + pair_acc_f[1]
  sum_surv_f = sum_surv_f + pair_acc_f[2]
}
print(paste("Logistic Regression: Avg Accuracy Survival", sum_surv_acc / 10))
print(paste("Logistic Regression: Avg Balanced Accuracy Survival", sum_surv_f / 10))

# Decision Tree
sum_surv_acc = 0
sum_surv_f = 0
for (t_set in survival_index){
  train = df[-t_set,]
  test = df[t_set,]
  
  # Performs better using all genes, rather than the best from Linear Regression
  dtree_survival = rpart(SURVIVAL~., data=train, method='class')
  pair_acc_f = evalr(dtree_survival, test, test$SURVIVAL, 'class')
  sum_surv_acc = sum_surv_acc + pair_acc_f[1]
  sum_surv_f = sum_surv_f + pair_acc_f[2]
}
print(paste("Decision Tree: Avg Accuracy Survival", sum_surv_acc / 10))
print(paste("Decision Tree: Avg Balanced Accuracy Survival", sum_surv_f / 10))

# Naive Bayes
# Iterate through validation sets
sum_surv_acc = 0
sum_surv_f = 0
for (t_set in survival_index){
  train = df[-t_set,]
  test = df[t_set,]
  
  # Best using all genes
  naive_bayes_survival = naiveBayes(SURVIVAL~., data=train, laplace=0)
  pair_acc_f = evalr(naive_bayes_survival, test, test$SURVIVAL, "class")
  sum_surv_acc = sum_surv_acc + pair_acc_f[1]
  sum_surv_f = sum_surv_f + pair_acc_f[2]
}
print(paste("NaiveBayes: Avg Accuracy Survival", sum_surv_acc / 10))
print(paste("NaiveBayes: Avg Balanced Accuracy Survival", sum_surv_f / 10))

# SVM
# Survival
for(kern in c("linear", "radial", "sigmoid", "polynomial")){
  # Iterate through validation sets
  sum_surv_acc = 0
  sum_surv_f = 0
  for (t_set in survival_index){
    train = df[-t_set,]
    test = df[t_set,]
    
    svm_model <- svm(formula=SURVIVAL~TP53BP1+CTBP1+APTX+RAD52+PMS1, kernel=kern, data=train, type="C-classification")
    pair_acc_f = evalr(svm_model, test, test$SURVIVAL, "class")
    sum_surv_acc = sum_surv_acc + pair_acc_f[1]
    sum_surv_f = sum_surv_f + pair_acc_f[2]
  }
  print(paste("SVM: Avg Accuracy Survival", kern, sum_surv_acc / 10))
  print(paste("SVM: Avg Balanced Accuracy Survival", kern, sum_surv_f / 10))
}
