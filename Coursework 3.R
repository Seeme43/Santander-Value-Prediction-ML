###### ============== P A C K A G E S   #######
library('readr')
library('dplyr')
library('finalfit') 
library('gbutils') 
library("Hmisc")
library("magrittr")
library('VIM')
library('mice')
library('corrplot')
library('caret')
library('psych')
library('Stack')
library('magrittr')
library('imputeMissings')
#library('tidyverse')
library('stats')
library('fastDummies')
library('aCRM')
library('DMwR')
library('glmnet')
library('car')
library('caret')
library('e1071')
library('schoolmath')
library('CORElearn')
library('AppliedPredictiveModeling')
library('pROC')
library('naniar')
library(readr)
library(ggplot2)
library(doParallel)
library(tidyverse)
library('randomForest')
library(plyr)
library(xgboost)
library(ggcorrplot)
library(FactoMineR)
library(factoextra)
library(paran)
library(resample)
library(Metrics)
library(matrixStats)
library(gbm)
library(ggthemes)
library(GGally)




###### ============== I M P O R T  T H E  D A T A  ######

train <- read_csv("train.csv")
test <-  read_csv('test.csv')


###### ============== A N A L Y S E  T H E  D A T A S E T  ######

#  Overview of data structure
str(train) 
str(test)
summary(train)[,1:10]

# Verify the Distribution of the Target Value

hist(train$target,  xlab='Value of the Output', 
     main='Distribution of the Target Variable', 
     col='coral')

# 1. Verify how many zeroes ( Percentage )
zeroes <- train %>% dplyr::select(-target, -ID)
sprintf("%.1f", sum(zeroes == 0 , na.rm = TRUE)/(nrow(zeroes) * ncol(zeroes))*100)# 96.9 % in Train data

# 2. Analysis of the rows for some costumers
transposed <- as.data.frame(t(train %>% dplyr::select(-target, -ID)))
str(transposed)

# Visualise the values for some different Rows
transposed$index <- seq(1:4991)
ggplot(transposed, aes(x = transposed$index, y= transposed$V30)) + 
          geom_line(color = "black") + labs(x = 'Values per Row / Client', y = 'Value')+ 
          theme_few()
ggplot(transposed, aes(x = transposed$index, y= transposed$V80)) + geom_line()
ggplot(transposed, aes(x = transposed$index, y= transposed$V140)) + geom_line()
ggplot(transposed, aes(x = transposed$index, y= transposed$V170)) + geom_line()

# Visualise the values for some different Columns
ggplot(train, aes(x = transposed$index[1:4459], y= train$`15d57abf7`)) + geom_line()
ggplot(train, aes(x = transposed$index[1:4459], y= train$b219e3635)) + geom_line()
ggplot(train, aes(x = transposed$index[1:4459], y= train$adf119b9a)) + geom_line()


# 3. Analysis of the Summary Statistics indexes
train_na = train %>% dplyr::select(-target, -ID) # Replace 0 with Nan's to exclude them calculations
train_na[train_na == 0] <- NA


# Training Data ( by Row !)
mean_train <- rowMeans(train_na, na.rm = TRUE)
std_train = apply(train_na, 1,  function(x) sd(x, na.rm = TRUE))
min_train = rowMins(as.matrix(train_na), na.rm = TRUE)
max_train = rowMaxs(as.matrix(train_na), na.rm = TRUE)
median_train = apply(train_na, 1,  function(x) median(x, na.rm = TRUE))
skewness_train = apply(train_na, 1,  function(x) skewness(x, na.rm = TRUE))
non_zero_train <- apply(train_na, 1 , function(x) sum(x != 0, na.rm = TRUE))
unique_values_train <- apply(train_na, 1, function(x) length(unique(x) != 'NA')-1) # To exclude NA's from the count

train_describe = data.frame('id' = train$ID ,'mean' = mean_train, 'standard_deviation' = std_train,
                            'non_zero_values' = non_zero_train, 'unique_non_zero' = unique_values_train,
                            'skewness' = skewness_train, 'median' = median_train, 'min' = min_train,
                            'max' = max_train, 'target' = train$target)

# Visualise the results
ggpairs(train_describe %>% dplyr::select(-id), title = 'Pair Plot') 
ggplot(train_describe, aes(x = non_zero_train , y= unique_non_zero)) + geom_point()+
  labs(x = 'Number of non zero values per Row', y = 'number of unique values per Row'
       )
  


# Test Data
test_na = test %>% dplyr::select(-ID) # I replace 0 with na in order to exclude them from my mean
test_na[test_na == 0] <- NA
mean_test <- rowMeans(test_na , na.rm = TRUE)
std_test = apply(test_na, 1,  function(x) sd(x, na.rm = TRUE))
min_test = rowMins(as.matrix(test_na), na.rm = TRUE)
max_test = rowMaxs(as.matrix(test_na), na.rm = TRUE)
median_test = apply(test_na, 1,  function(x) median(x, na.rm = TRUE))
skewness_test = apply(test_na, 1,  function(x) skewness(x, na.rm = TRUE))
non_zero_test <- apply(test_na, 1 , function(x) sum(x != 0, na.rm = TRUE))
unique_values_test <- apply(test_na, 1, function(x) length(unique(x) != 'NA') -1 ) # To exclude NA's from the count

test_describe = data.frame('id' = test$ID,  'mean' = mean_test, 'standard_deviation' = std_test,
                           'non_zero_values' = non_zero_test, 'unique_non_zero' = unique_values_test,
                           'skewness' = skewness_test, 'median' = median_test,'min' = min_test,
                           'max' = max_test)

# Visualise the Results 
pairs(test_describe)
ggplot(test_describe, aes(x = non_zero_values, y= unique_non_zero)) + geom_point()


# Artificial Noise ? Check how much artificial noise is present in here:
nrow(train_describe[train_describe$non_zero_values == train_describe$unique_non_zero,])
nrow(test_describe[test_describe$non_zero_values == test_describe$unique_non_zero,])

# Store which rows to eventually use them if you need it:
#noisy_column_test = test_describe[test_describe$non_zero_values == test_describe$unique_non_zero,]$id # delete when PCA 
noisy_column_train = train_describe[train_describe$non_zero_values == train_describe$unique_non_zero,]
noisy_column_train = noisy_column_train$id


###### ============== D A T A  C L E A N I N G    ########

# 1. Randomise the Train dataset, before Joining it 
set.seed(1)
train_ <- train[sample(nrow(train)),] # risky situation cause it may be a time series, but I assume that it can eventually be a time series by row.
                                     # I can always go back after and make other assumptions


# 2. Join Train and Test data to analyse the structure
target <- train_$target
train_X <- train_
train_X$target <- NULL
all_dataX <- rbind(train_X, test)


# 3. Modify the Variables type, if needed
# Verify how many numerical and Categorical variables I have
length(data_num <- dplyr::select_if( all_dataX , is.numeric)) # How many numerical variables
length(data_chr <- dplyr::select_if(all_dataX, is.character)) # How many categorical

# Investigate which one is the categorical variable
nrow(unique(data_chr)) # Number of Uniques values 

# Remove id column 
all_dataX$ID <- NULL
train_X$ID <- NULL



###### ============== D I M E N S I O N A L I T Y  R E D U C T I O N  #########

# 0. MISSING VALUES 
isna_ <- all_dataX %>% dplyr::summarise_all(funs(sum(is.na(.)))) %>% gather(variable, num_NA) 
#View(isna_ <- filter(isna_, isna_$num_NA > 0)) # Visualise the number of Nan's per column
isna_ <- filter(isna_, isna_$num_NA > 0) # Keep only the columns with Nan's
na_columns <- isna_$variable # Store the name of the columns to apply imputation
na_alldata <- all_dataX[, na_columns]

mean = colMeans(na_alldata, na.rm = TRUE) # Replace with Mean with a For Loop
for(i in 1:ncol(na_alldata)){
  na_alldata[is.na(na_alldata[,i]), i] <- mean[[i]]
}


# Delete the columns on the all_dataX and put the imputed ones
all_dataX <- all_dataX[,!(names(all_dataX) %in% na_columns)]
all_dataX <- cbind(all_dataX, na_alldata)



# 1. NO VARIANCE ( Variance = 0) 

# Create a Trainset(70% of Train_X, against 30% --> Valset)
trainset <- all_dataX[1:3200,]
# valset <- all_dataX[3201:4459,] # No needed so far

# Verify how many unique values per Variable
unique_ <- trainset %>% dplyr::summarise_all(funs(length(unique(.)))) %>% gather(variable, unique) 
unique_ <- filter(unique_, unique_$unique == 1) # only those with 1 value ( Variance = 0)
zerovar = unique_$variable # Store the name of the ZeroVariance Variables
all_dataX <- all_dataX[, !names(all_dataX) %in% zerovar] # Remove ZeroVariance Variables




# 2. LOW VARIANCE ( very conservative ) 

# Set my laptop in order to use Parallel Computing 
detectCores()
cl <- makeCluster(detectCores() - 2)
clusterEvalQ(cl, library(foreach))
registerDoParallel(cl)   # register this cluster
getDoParWorkers()

 
trainset <- all_dataX[1:3200,] # Create again Trainset ( from the updated All_DataX)
nzv <- nearZeroVar(trainset, saveMetrics = TRUE, allowParallel = TRUE) # Save Metrics to Verify the distribution
print(paste('FreqRatio:', range(nzv$freqRatio))) 
print(paste('Range:', range(nzv$percentUnique)))


# Graphic Visualisation 
hist(nzv$freqRatio, freq=FALSE, xlab='FreqCut', 
     main='Distribution of FreqCut parameter ', 
     col='coral')
hist(nzv$percentUnique, freq=FALSE, xlab='UniqueCut', 
     main='Distribution of UniqueCut parameter over all Variables', 
     col='lightgreen')

hist(nzv$freqRatio) # graphically visualise 
hist(nzv$percentUnique)

head(nzv)
dim(nzv[nzv$zeroVar, ]) # Is there still some with Zero Variance ?
dim(nzv[nzv$percentUnique < 0.1, ])

# Apply a conservative reduction of Low Variance variables
nearvar <- nearZeroVar(trainset, freqCut = 3000, uniqueCut = 0.1 , allowParallel = TRUE)
all_dataX <- all_dataX[, - nearvar]


################################### 150 high Variance Columns ################################################ 
# Keep 150 Variables with Most Variance for future model building 
dim(nzv[nzv$percentUnique >= 11  & nzv$freqRatio < 110,]) 
highvar <- nearZeroVar(trainset, freqCut = 110, uniqueCut = 11 , allowParallel = TRUE)
# highvar_alldata <- all_dataX[, - highvar]


# 3. CORRELATION BETWEEN VARIABLES ? 
trainset <- all_dataX[1:3200,]
length(data_num <- dplyr::select_if( trainset , is.numeric)) # How many numerical variables

trainset <- as.matrix(trainset)
corr_data <- cor(trainset, method = 'spearman') 
corr_var <- findCorrelation(corr_data, cutoff=0.7, names=TRUE)  
corr_var <- sort(corr_var)

#Removing all variables with a correlation of over .7 
all_dataX <-all_dataX[, !(colnames(all_dataX) %in% c(corr_var))]


########## ========== D A T A  T R A N S F O R M A T I O N  #########

# 1. S k e w n e s s 

# Skewness test ( check if working on train only )
trainset <- all_dataX[1:3200,]
(skew_train_pre = apply(trainset, 2, function(x) skewness(x, na.rm = TRUE)))
(skew_tot_pre = apply(all_dataX, 2, function(x) skewness(x, na.rm = TRUE)))


# BoxCox 
trainset = trainset + 1
BoxCox = preProcess(trainset, method = 'BoxCox', verbose = TRUE) #, na.remove = TRUE)
all_dataX <- all_dataX + 1
all_dataX = predict(BoxCox, all_dataX)

# Verify if the Process worked
(skew_tot_post = apply(all_dataX, 2, function(x) skewness(x, na.rm = TRUE)))

skew_tot_pre[1:10]
skew_tot_post[1:10] # It worked 

# 2. S t a n d a r d i z a t i o n ( No big downside except for interpretability ) 
trainset <- all_dataX[1:3200,]
standardz = trainset %>% preProcess( method = c('scale','center'))
all_dataX = predict(standardz, all_dataX)

# 3. O u t l i e r s ( from package 'The predictors should be centered and scaled before applying this transformation.')
transformed = spatialSign(all_dataX)
all_dataX <- as.data.frame(transformed)

# Summary of the new dataset
summary(all_dataX[,10:20])


save.image('Pre-FeatureSelection')
load.image('Pre-FeatureSelection')

########## ========= F E A T U R E  S E L E C T I O N  #######


# 1. S u m m a r y  S t a t i s t i c s  P a r a m e t e r s  ####

# Pre processing of the Summary Statistics

# _Imputation of Missing Values 
isna_describe <- rbind(train_describe[,1:9], test_describe) %>% 
                 dplyr::summarise_all(funs(sum(is.na(.)))) %>% 
                 gather(variable, num_NA) 
isna_describe <- filter(isna_describe, isna_describe$num_NA > 0)
na_describe_columns <- isna_describe$variable
na_alldata <- rbind(train_describe[,1:9], test_describe)[, na_describe_columns]

mean = colMeans(train_describe[1:3200, na_describe_columns], na.rm = TRUE) # only on the training data
for(i in 1:ncol(na_alldata)){
  na_alldata[is.na(na_alldata[,i]), i] <- mean[[i]]
}

all_data_describe = rbind(train_describe[,1:9], test_describe)
all_data_describe <- all_data_describe[,!(names(all_data_describe) %in% na_describe_columns)]

all_data_describe <- cbind(all_data_describe, na_alldata)
train_describe <- cbind(all_data_describe[1:4459,-1], 'target' = train_$target)
test_describe <- cbind(all_data_describe[4459:53801,-1])


# _Correlation 

describe_train_matrix <- as.matrix(train_describe[1:3200,])
corr_data <- cor(describe_train_matrix) # <- Specify about the different correlation !!
corr_p <- rcorr(describe_train_matrix)

corrplot(corr_data, method="color", type='lower', p.mat=corr_p$P, sig.level=.05) # Correlation plot that also shows which correlations are significant
corr_var <- findCorrelation(corr_data, cutoff=0.75, names=TRUE)  
corr_var <- sort(corr_var)
corr_var

train_describe <- train_describe[, !names(train_describe) %in% corr_var]
test_describe  =   test_describe[, !names(test_describe) %in% corr_var]
#str(train_describe)


# Skewness
train_describeX <- train_describe[1:3200,] %>% dplyr::select(-target)
train_describeX = train_describeX + 1

(skew_describe_pre = apply(train_describeX, 2, function(x) skewness(x, na.rm = TRUE))) 
BoxCox_describe = preProcess(train_describeX, method = 'BoxCox', verbose = TRUE) #, na.remove = TRUE)
train_describe = predict(BoxCox_describe, train_describe %>% dplyr::select(-target))
train_describe = cbind(train_describe, 'target' = train_$target)
# test_describe = predict(BoxCox_describe, test_describe)
(skewValues = apply(train_describe, 2, function(x) skewness(x, na.rm = TRUE))) # Verify if the Process worked

# Standardization 
train_describeX <- train_describe[1:3200,] %>% dplyr::select(-target)
standardiz_describe = train_describeX %>% preProcess( method = c('scale','center'))
train_describe = predict(standardiz_describe, train_describe)
# test_describe = predict(standardiz_describe, test_describe)

# Outliers
transformed_describe = spatialSign(train_describe[,-5])
train_describe <- as.data.frame(transformed_describe)
train_describe = cbind(train_describe, 'target' = train_$target)

describe_train = train_describe[1:3200,]
describe_val = train_describe[3201:4459,]



#  BOOSTING 
boost_describe = gbm(target ~., data= describe_train, distribution= "gaussian", 
                 n.trees=5000, interaction.depth = 4, 
                 shrinkage = 0.01, verbose = F)
summary(boost_describe)

# Use the boosted model to predict label on the val set:
ydes.boost = predict(boost_describe, newdata = describe_val, n.trees=5000)

#ModelMetrics::rmsle(boost_hvar)
(nrmse_desc_boost = hydroGOF::nrmse(ydes.boost , describe_val$target))
(rmsle_desc_boost = ModelMetrics::rmsle(abs(describe_val$target), abs(ydes.boost)))


# XGBOOST

#desc_train.x <- xgb.DMatrix(as.matrix(describe_train %>% dplyr::select(-target))) #, missing = 0)
#desc_train.y <- describe_train$target
#desc_val.x <- xgb.DMatrix(as.matrix(describe_val %>% dplyr::select(-target)))
#desc_val.y <- describe_val$target


fourStats <- function (data, lev = NULL, model = NULL) {
  library(Metrics)
  out <- Metrics::rmsle(data[, "obs"], data[, "pred"])
  names(out) = c('rmsle')
  out
}

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = TRUE,
  summaryFunction = fourStats,
  returnData = FALSE
)

# This is the grid space to search for the best hyperparameters
xgbGrid <- expand.grid(nrounds = c(100,200, 300), 
                       max_depth = c(10, 15, 20),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = c(0.1),
                       gamma=c(0.1),
                       min_child_weight = c(1),
                       subsample = c(0.8) # default values - unchanged 
)

# Train the Algorithm
set.seed(0) 
xgb_model_describe = train( target ~., data = 
                     describe_train,
                   method = "xgbTree",
                   maximize = FALSE,
                   metric = 'rmsle',
                   tuneGrid = xgbGrid,
                   trControl = xgb_trcontrol
)

xgb_model_describe$bestTune # look for the best tune
varImp(xgb_model_describe) # Most important variables
ydescribe.xgboost = predict(xgb_model_describe, newdata = describe_val )
(nrmse_desc_xgboost = hydroGOF::nrmse(ydescribe.xgboost, describe_val$target))
(rmsle_desc_xgboost = ModelMetrics::rmsle(abs(describe_val$target), abs(ydescribe.xgboost)))



# 2.   1 5 0   V a r i a b l e s  w i t h   M o s t   V a r i a n c e ####


highvar_alldata <- all_dataX[, - highvar]
highvar_alldata <- cbind(highvar_alldata[1:"4459",], 'target' = target)
hvar_train = highvar_alldata[1:3200,]
hvar_val = highvar_alldata[3201:4459,]

#View(highvar_alldata[1:10,1:20])

# LINEAR REGRESSION with MIXED METHOD

lmAll <- lm( target ~ . , data = hvar_train )
lmNone <- lm(target ~ 1, data = hvar_train)
step(lmNone, direction = 'both', scope = formula(lmAll))
#summary(lmAll)

# StepForward + p-values feature reduction 
lm_hvar <- lm(target ~ `678e2d1dd` + dfde54714 + `07cb6041d` + 
                `540208409` + ea397d576 + `018ab6a80` + c906cd268 + `1c71183bb` + 
                `5fe6867a4` + `9d5c7cb94` + cdfc2b069 + `5adfe7419` + ce999e374 + 
                `1a7de209c` + `8ca08456c` + adc721d55 + `11d86fa6a` + `56b9c3eb3` + 
                `643ef6977` + `6045a2949` + c90b0b8a7 + d3ed79990 + `5719bbfc3` , 
              data = hvar_train)
summary(lm_hvar) 
yhvar.lin = predict(lm_hvar, newdata = hvar_val)
(nrmse_hvar_lm = hydroGOF::nrmse(yhvar.lin , hvar_val$target))

lm_hvar <- lm(target ~ `678e2d1dd` + dfde54714 + `07cb6041d` + 
                `540208409`  + c906cd268 , 
              data = hvar_train)
summary(lm_hvar) 
yhvar.lin = predict(lm_hvar, newdata = hvar_val)
(nrmse_hvar_lm = hydroGOF::nrmse(yhvar.lin , hvar_val$target))
(rmsle_hvar_lm = ModelMetrics::rmsle(abs(hvar_val$target), abs(yhvar.lin)))


# BOOSTING

# you have to do cross validation by yourself to calculate the shrinking parameter etc..
boost_hvar = gbm(target~.,data= hvar_train, distribution= "gaussian", 
                 n.trees=5000, interaction.depth=4, 
                 shrinkage = 0.01, verbose = F)
summary(boost_hvar)

# We now use the boosted model to predict medv on the test set:
yhvar.boost = predict(boost_hvar, newdata = hvar_val, n.trees=5000)


#ModelMetrics::rmsle(boost_hvar)
(nrmse_hvar_boost = hydroGOF::nrmse(yhvar.boost , hvar_val$target))
(rmsle_hvar_boost = ModelMetrics::rmsle(abs(hvar_val$target), abs(yhvar.boost)))


# XGBOOST ( Tried to use DMatrix for the XGBoost but returning me error)
#hvar_train.x <- xgb.DMatrix(as.matrix(hvar_train %>% dplyr::select(-target))) #, missing = 0)
#hvar_train.y <- hvar_train$target
#hvar_val.x <- xgb.DMatrix(as.matrix(hvar_val %>% dplyr::select(-target)))
#hvar_val.y <- hvar_val$target

set.seed(0) 
xgb_model = train( target ~., data = 
  hvar_train,
  method = "xgbTree",
  maximize = FALSE,
  metric = 'rmsle',
  tuneGrid = xgbGrid,
  trControl = xgb_trcontrol
)

xgb_model$bestTune
varImp(xgb_model)
yhvar.xgboost = predict(xgb_model, newdata = hvar_val)
(nrmse_hvar_xgboost = hydroGOF::nrmse(yhvar.xgboost, hvar_val$target))
(rmsle_hvar_xgboost = ModelMetrics::rmsle(abs(hvar_val$target), abs(yhvar.xgboost)))


##### 3. P r i n c i p a l  C o m p o n e n t  A n a l y s i s  ##### 


testdata <- all_dataX[4450:53801,]
all_train_data <- cbind(all_dataX[1:4459,], 'target' = target)
trainset = all_train_data[1:3200,]
valset = all_train_data[3201:4459,]
x.train <- trainset %>% dplyr::select(-target) # to build the PCA

# Define the PCA
pca_output <- PCA( x.train , ncp = 100, graph = FALSE)

# Get the summary of the first 10 variables
summary(pca_output, nbelements = 10)

# Get the variance of the first 20 new dimensions.
pca_output$eig[,2][1:20]
singlevar = pca_output$eig[,2] # Store the variance into a variable to plot it 

# Get the cumulative variance. 
pca_output$eig[,3][1:50]
cumul = pca_output$eig[,3]  # Store the cumulative variance into a variable to plot it 
plot(cumul, type= 'l', col="red" )
par(new=TRUE)
plot(singlevar, type="l", col="blue") 
pca_output$eig[,3][1:100] # variance 


# Get the most correlated variables
# dimdesc(pca_output, axes = 1:2)
# Contribution of the variables 
#pca_output$var$contrib

# Create a factor map for the top 5 variables with the highest contributions.
fviz_pca_var(pca_output, select.var = list(contrib = 5), repel = TRUE)

# Create a barplot for the variables with the highest contributions to the 1st PC.
#fviz_contrib(pca_output, choice = "var", axes = 1, top = 5)



# Define How Many Variables to Keep  

# 0. Heuristic Approach
pca_output$eig[,3][as.integer(pca_output$eig[,3]) == 85] # <- Heuristic Approach to define the number of component to keep 
# 1298 dimensions 


# 1. Scree Test ( Elbow criteria )
# Perform the screeplot test
fviz_screeplot(pca_output, ncp = 10)


# 2. Keyser-Guttman Rule ( Keep the PCA with Eigenvalue > 1)
get_eigenvalue(pca_output)[1170:1195]
get_eigenvalue(pca_output)[1192:1193] # 1193 dimensions


# 3. Parallel Analysis 
#library(paran)
#air_paran <- paran(trainset, seed = 1)
# Check out the suggested number of PCs to retain.
#air_paran$Retained
# Conduct a parallel analysis with fa.parallel().
#air_fa_parallel <- fa.parallel(airquality)
# Check out the suggested number of PCs to retain.
#air_fa_parallel$ncomp


# Obtain the PCA 
pca <- preProcess(x.train, method = 'pca')
pca_data <- predict(pca, all_dataX)

pca_data_train <- pca_data[1:3200, 1:1240] # with 1847 you describe 95 % 
pca_data_train <- cbind(pca_data_train, 'target' = target[1:3200])
pca_data_val <- pca_data[3201:4459, 1:1240]
pca_data_val <- cbind(pca_data_val, 'target' = target[3201:4459])

save.image('postPCA')
#load.image('postPCA')

# LINEAR REGRESSION 
lmAll_pca <- lm( target ~ . , data = pca_data_train )
lmNone_pca <- lm(target ~ 1, data = pca_data_train)
# step_pca = step(lmNone_pca, direction = 'both', scope = formula(lmAll_pca)) Too expensive fo the result 
forw_pca = step(lmNone_pca, direction = 'forward', scope = formula(lmAll_pca))
summary(forw_pca)
# Turned out to be too computationally expensive for the result it provides

# Forward + Feature based on p-values
forw_pca <- lm(target ~ PC3 + PC39 + PC38 + PC13 + PC50 + PC26 + 
               PC34 + PC6 + PC11 + PC43 + PC84 + PC70  + 
               PC22 + PC56  + PC2 + PC35 + 
               PC18 + PC1 + PC62  + PC42 + PC1230 + PC1228 + PC51 + PC8  + PC143 + 
                PC24 + PC919 + PC29  + PC668 + 
               PC196, data = pca_data_val)
summary(forw_pca) 

ypca.lin = predict(forw_pca, newdata = pca_data_val)
(nrmse_pca_lm = hydroGOF::nrmse(ypca.lin , pca_data_val$target))
(rmsle_pca_lm = ModelMetrics::rmsle(abs(pca_data_val$target), abs(ypca.lin)))


# BOOSTING 
boost_pca = gbm(target~.,data= pca_data_train, distribution= "gaussian", 
                 n.trees=5000, interaction.depth=4, 
                 shrinkage = 0.01, verbose = F)
summary(boost_pca)
ypca.boost = predict(boost_pca, newdata = pca_data_val, n.trees=5000)
#ModelMetrics::rmsle(boost_hvar)
(nrmse_pca_boost = hydroGOF::nrmse(ypca.boost , pca_data_val$target))
(rmsle_pca_boost = ModelMetrics::rmsle(abs(pca_data_val$target), abs(ypca.boost)))

# X G B O O S T I N G 

set.seed(0) 
xgb_model_pca = train( target ~., data = 
                    pca_data_train,
                    method = "xgbTree",
                    maximize = FALSE,
                    metric = 'rmsle',
                    tuneGrid = xgbGrid,
                    trControl = xgb_trcontrol
)

xgb_model_pca$bestTune
ypca.xgboost = predict(xgb_model_pca, newdata = pca_data_val)
(nrmse_pca_xgboost = hydroGOF::nrmse(ypca.xgboost, pca_data_val$target))
(rmsle_pca_xgboost = ModelMetrics::rmsle(abs(pca_data_val$target), abs(ypca.xgboost)))

range(xgb_model_pca$results$rmsle, na.rm = TRUE)
hist(xgb_model_pca$results$rmsle)


varImp(xgb_model_pca)


# 4. L a s s o  R e g r e s s i o n  #####

x.train <- as.matrix(trainset %>% dplyr::select(-target))
y.train <- as.matrix(trainset %>% dplyr::select(target))
x.val <- as.matrix(valset %>% dplyr::select(-target))
y.val <- as.matrix(valset %>% dplyr::select(target))

cv.out = cv.glmnet(x.train, y.train, alpha = 1, parallel = TRUE )  # alpha = 1 means Lasso 
plot(cv.out)
bestlam=cv.out$lambda.min

lasso.mod=glmnet(x.train, y.train, alpha=1, lambda= bestlam)
plot(lasso.mod)

y.lasso = predict(lasso.mod,s=bestlam ,newx=x.val)
(nrmse_lasso = hydroGOF::nrmse(y.lasso, y.val))
(rmsle_lasso = ModelMetrics::rmsle(abs(y.val), abs(y.lasso)))

lasso.coeff = predict(lasso.mod, type = 'coefficients' , newx = x.val)
length(lasso.coeff[lasso.coeff != 0])
lasso_var = coef(lasso.mod)


# Keep only variables selected by Lasso 
lasso_alldataX = as.data.frame(all_dataX[, lasso_var@Dimnames[[1]][lasso_var@i][-1]]) # https://stackoverflow.com/questions/43785750/after-lasso-store-remaining-variables-as-new-dataframe-using-r
lasso_alldata <- cbind(lasso_alldataX[1:4459,], 'target' = target)
lasso_train = lasso_alldata[1:3200,]
lasso_val = lasso_alldata[3201:4459,]


# LINEAR REGRESSION 

lmAll_lasso <- lm( target ~ . , data = lasso_train )
lmNone_lasso <- lm(target ~ 1, data = lasso_train)
forw_lasso = step(lmNone_lasso, direction = 'forward', scope = formula(lmAll_lasso))
summary(forw_lasso)


# Forward + Feature based on p-values
forw_lasso <- lm(formula = target ~ fcda960ae + aeff360c7 + `77eb013ca` + d6af4ee1a + 
                 `4b9540ab3` + `83e2ae51c` + `8d4f4c571` + fa977f17b + `698d05d29` + 
                 `05cc08c11` + `8675bec0b` + `468d2c3b6` + `616be0c3e` + d8296080a + 
                 `35da68abb` + `423058dba` + b94360a3b + `59d2470ed` + a636266f3 + 
                 `982210169` + `5f11fbe33` + `47665e3ce` + `8a1b76aaf` + `902c5cd15` + 
                 b3abb64d2 + `0e3ef9e8f`, data = lasso_train)
summary(forw_lasso) 

ylasso.lin = predict(forw_lasso, newdata = lasso_val)
(nrmse_lasso_lm = hydroGOF::nrmse(ylasso.lin , lasso_val$target))
(rmsle_lasso_lm = ModelMetrics::rmsle(abs(lasso_val$target), abs(ylasso.lin)))


# BOOSTING 
boost_lasso = gbm(target~.,data= lasso_train, distribution= "gaussian", 
                n.trees=5000, interaction.depth=4, 
                shrinkage = 0.01, verbose = F)
summary(boost_lasso)
ylasso.boost = predict(boost_lasso, newdata = lasso_val, n.trees=5000)

(nrmse_lasso_boost = hydroGOF::nrmse(ylasso.boost , lasso_val$target))
(rmsle_lasso_boost = ModelMetrics::rmsle(abs(lasso_val$target), abs(ylasso.boost)))

# X G B O O S T I N G 

set.seed(0) 
xgb_model_lasso = train( target ~., data = 
                           lasso_train,
                       method = "xgbTree",
                       maximize = FALSE,
                       metric = 'rmsle',
                       tuneGrid = xgbGrid,
                       trControl = xgb_trcontrol
)

xgb_model_lasso$bestTune
ylasso.xgboost = predict(xgb_model_lasso, newdata = lasso_val)
(nrmse_lasso_xgboost = hydroGOF::nrmse(ylasso.xgboost, lasso_val$target))
(rmsle_lasso_xgboost = ModelMetrics::rmsle(abs(lasso_val$target), abs(ylasso.xgboost)))



# 5. E l a s t i c  N e t  #####
#( Avoid the correlation function here cause it's handled by Elastic net )
cv.out.net = cv.glmnet(x.train, y.train, alpha=0.5, parallel = TRUE ) 
plot(cv.out.net)
bestlam=cv.out.net$lambda.min
#coef(cv.out.net, s= bestlam)

net.mod=glmnet(x.train, y.train, alpha=0.5, lambda= bestlam)
plot(net.mod)

 
y.net = predict(net.mod, s=bestlam ,newx=x.val)
(nrmse_net = hydroGOF::nrmse(y.net, y.val))
(rmsle_net = ModelMetrics::rmsle(abs(y.val), abs(y.net)))

net.coeff = predict(net.mod, type = 'coefficients' , newx = x.val)
length(net.coeff[net.coeff != 0])
net_var = coef(net.mod)


# Keep only variables selected by Lasso 
net_alldataX = as.data.frame(all_dataX[, net_var@Dimnames[[1]][net_var@i][-1]]) 

# How many values in Common ?
length(intersect(names(net_alldataX),names(lasso_alldataX)))



# 6. L a s s o  W i t h  P C A 

pca.x.train <- as.matrix(pca_data_train %>% dplyr::select(-target))
pca.y.train <- as.matrix(pca_data_train %>% dplyr::select(target))
pca.x.val <- as.matrix(pca_data_val %>% dplyr::select(-target))
pca.y.val <- as.matrix(pca_data_val %>% dplyr::select(target))


cv.out.pca = cv.glmnet(pca.x.train, pca.y.train, alpha=0.5, parallel = TRUE ) 
plot(cv.out.pca)
bestlam=cv.out.pca$lambda.min
#coef(cv.out.net, s= bestlam)

pca.mod=glmnet(pca.x.train, pca.y.train, alpha=0.5, lambda= bestlam)
plot(pca.mod)


y.pca_net = predict(pca.mod, s=bestlam ,newx = pca.x.val)
(nrmse_pca_net = hydroGOF::nrmse(y.pca_net, pca.y.val))
(rmsle_pca_net = ModelMetrics::rmsle(abs(pca.y.val), abs(y.pca_net)))



pca.net.coeff = predict(pca.mod, type = 'coefficients' , newx = pca.x.val)
length(pca.net.coeff[pca.net.coeff != 0])
pca_net_var = coef(pca.mod)

# Keep only variables selected by Lasso 
pca_net_alldataX = as.data.frame(pca_data[, pca_net_var@Dimnames[[1]][pca_net_var@i][-1]]) 

pca_net_alldata <- cbind(pca_net_alldataX[1:4459,], 'target' = target)
pca_net_train = pca_net_alldata[1:3200,]
pca_net_val =  pca_net_alldata[3201:4459,]




# BOOSTING 
boost_pca_net = gbm(target~.,data= pca_net_train, distribution= "gaussian", 
                  n.trees=5000, interaction.depth=4, 
                  shrinkage = 0.01, verbose = F)
summary(boost_pca_net)
ypca_net.boost = predict(boost_pca_net, newdata = pca_net_val, n.trees=5000)

(nrmse_pca_net_boost = hydroGOF::nrmse(ypca_net.boost , pca_net_val$target))
(rmsle_pca_net_boost = ModelMetrics::rmsle(abs(pca_net_val$target), abs(ypca_net.boost)))



# X G B O O S T I N G 

set.seed(0) 
xgb_model_pca_net = train( target ~., data = 
                           pca_net_train,
                         method = "xgbTree",
                         maximize = FALSE,
                         metric = 'rmsle',
                         tuneGrid = xgbGrid,
                         trControl = xgb_trcontrol
)

xgb_model_pca_net$bestTune
ypca_net.xgboost = predict(xgb_model_pca_net, newdata = pca_net_val)
(nrmse_pca_net_xgboost = hydroGOF::nrmse(ypca_net.xgboost, pca_net_val$target))
(rmsle_pca_net_xgboost = ModelMetrics::rmsle(abs(pca_net_val$target), abs(ypca_net.xgboost)))

range(xgb_model_pca_net$results$rmsle, na.rm = TRUE)
hist(xgb_model_pca_net$results$rmsle)






# 6. C o m b i n e  T h e  R e s u l t s  ####

# I try to check if adding some summary statistics may add value to the model 
pca_desc_train = cbind(pca_net_train, describe_train[,1:4]) 
pca_desc_train$standard_deviation <- NULL # st. deviation had no importance so I will remove it
pca_desc_val = cbind(pca_net_val, describe_val[,1:4]) 
pca_desc_val$standard_deviation <- NULL



set.seed(0) 
xgb_model_pca_desc = train( target ~., data = 
                             pca_desc_train,
                           method = "xgbTree",
                           maximize = FALSE,
                           metric = 'rmsle',
                           tuneGrid = xgbGrid,
                           trControl = xgb_trcontrol
)

xgb_model_pca_desc$bestTune
ypca_desc.xgboost = predict(xgb_model_pca_desc, newdata = pca_desc_val)
(nrmse_pca_desc_xgboost = hydroGOF::nrmse(ypca_desc.xgboost, pca_desc_val$target))
(rmsle_pca_desc_xgboost = ModelMetrics::rmsle(abs(pca_desc_val$target), abs(ypca_desc.xgboost)))

range(xgb_model_pca_desc$results$rmsle, na.rm = TRUE)
hist(xgb_model_pca_desc$results$rmsle)

varImp(xgb_model_pca_desc)[1:50]

# It improves the result but not much, and those variables do not seem to be very important in it


# 7. K e e p  O n l y  50  P C A  f r o m  X G B #####

var_50_xgb = varImp(xgb_model_pca_desc)
var_50_xgb = rownames(var_50_xgb$importance)[1:50]


pca_50_alldataX = as.data.frame(pca_data[, (colnames(pca_data) %in% c(var_50_xgb))]) 
pca_50_alldata <- cbind(pca_50_alldataX[1:4459,], 'target' = target)
pca_50_train = pca_50_alldata[1:3200,]
pca_50_val =  pca_50_alldata[3201:4459,]


# XGB
set.seed(0) 
xgb_model_pca_50 = train( target ~., data = 
                            pca_50_train,
                           method = "xgbTree",
                           maximize = FALSE,
                           metric = 'rmsle',
                           tuneGrid = xgbGrid,
                           trControl = xgb_trcontrol
)

xgb_model_pca_50$bestTune
ypca_50.xgboost = predict(xgb_model_pca_50, newdata = pca_50_val)
(nrmse_pca_50_xgboost = hydroGOF::nrmse(ypca_50.xgboost , pca_50_val$target))
(rmsle_pca_50_xgboost = ModelMetrics::rmsle(abs(pca_50_val$target), abs(ypca_50.xgboost )))


range(xgb_model_pca_50$results$rmsle, na.rm = TRUE)
hist(xgb_model_pca_50$results$rmsle)


# BOOSTING 
boost_pca_50 = gbm(target~.,data= pca_50_train, distribution= "gaussian", 
                    n.trees=5000, interaction.depth=4, 
                    shrinkage = 0.01, verbose = F)
summary(boost_pca_50)
ypca_50.boost = predict(boost_pca_50, newdata = pca_50_val, n.trees=5000)

(nrmse_pca_50_boost = hydroGOF::nrmse(ypca_50.boost , pca_50_val$target))
(rmsle_pca_50_boost = ModelMetrics::rmsle(abs(pca_50_val$target), abs(ypca_50.boost)))





# P C A  s e e m s  T h e  R i g h t  R o a d


##########    M O D E L  B U I L D I N G ####

# Let's try to optimize the model keeping only 50 PCA, but first, what if I keep 80 ?

# A. T r y 80 P C A s #####

var_80_xgb = varImp(xgb_model_pca_net)  # From the PCA feature selection with Elastic NET
var_80_xgb = rownames(var_80_xgb$importance)[1:50]


pca_80_alldataX = as.data.frame(pca_data[, (colnames(pca_data) %in% c(var_80_xgb))]) 
pca_80_alldata <- cbind(pca_80_alldataX[1:4459,], 'target' = target)
pca_80_train = pca_80_alldata[1:3200,]
pca_80_val =  pca_80_alldata[3201:4459,]


# XGB 
set.seed(0) 
xgb_model_pca_80 = train( target ~., data = 
                            pca_80_train,
                          method = "xgbTree",
                          maximize = FALSE,
                          metric = 'rmsle',
                          tuneGrid = xgbGrid,
                          trControl = xgb_trcontrol
)

xgb_model_pca_80$bestTune
ypca_80.xgboost = predict(xgb_model_pca_80, newdata = pca_80_val)
(nrmse_pca_80_xgboost = hydroGOF::nrmse(ypca_80.xgboost , pca_80_val$target))
(rmsle_pca_80_xgboost = ModelMetrics::rmsle(abs(pca_80_val$target), abs(ypca_80.xgboost )))

range(xgb_model_pca_80$results$rmsle, na.rm = TRUE)
hist(xgb_model_pca_80$results$rmsle)





# B. Linear Regression to Give Interpretability of the Data ####

lmAll_pca_50 <- lm( target ~ . , data = pca_50_train )
lmNone_pca_50 <- lm(target ~ 1, data = pca_50_train)
forw_pca_50 = step(lmNone_pca_50, direction = 'forward', scope = formula(lmAll_pca_50))

summary(forw_pca_50)



# Forward + Feature based on p-values
forw_pca_50 <- lm(formula = target ~ PC3 + PC39 + PC38 + PC7 + PC34 + PC6 + 
                       PC11 + PC43 + PC84 + PC36 + PC37 + PC25 + PC12 + PC2 + PC35 + 
                       PC1 + PC67 + PC4 + PC186 + PC8 + PC971 + PC28, data = pca_50_train)

summary(forw_pca_50) 

ypca_50.lin = predict(forw_pca_50, newdata = pca_50_val)
(nrmse_pca_50_lm = hydroGOF::nrmse(ypca_50.lin , pca_50_val$target))
(rmsle_pca_50_lm = ModelMetrics::rmsle(abs(pca_50_val$target), abs(ypca_50.lin)))



##### C. X g b  B o o s t i n g   H y p e r P a r a m e t e r s  ######

# Can be best when you have so much variables 

set.seed(0) 


xgb_trcontrol = trainControl(
  method = "cv",
  number = 10,  
  allowParallel = TRUE,
  verboseIter = TRUE,
  returnData = TRUE,
  classProbs = TRUE,
  search = "random",
  savePredictions = "final"
  ,summaryFunction = fourStats
)


xgbGrid <- expand.grid(nrounds = c(100,200,300), # max number of trees to build 
                       max_depth = c(5, 10, 15, 20),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = c(0.1,0.01), # learning rate
                       gamma=c(0.1),
                       min_child_weight = c(1),
                       subsample = c(0.8) 
)




set.seed(0) 
xgb_model_hyper = train( target ~., data = 
                            pca_50_train,
                          method = "xgbTree",
                          maximize = FALSE,
                          metric = 'rmsle',
                          tuneGrid = xgbGrid,
                          trControl = xgb_trcontrol
)

xgb_model_hyper$bestTune
yhyper.xgboost = predict(xgb_model_hyper, newdata = pca_50_val)
(nrmse_hyper_xgboost = hydroGOF::nrmse(yhyper.xgboost , pca_50_val$target))
(rmsle_hyper_xgboost = ModelMetrics::rmsle(abs(pca_50_val$target), abs(yhyper.xgboost )))

range(xgb_model_hyper$results$rmsle, na.rm = TRUE)
hist(xgb_model_hyper$results$rmsle)

xgb_model_hyper$results[10:20,]


# Attempt 2 to tune hyperparameters


xgbGrid <- expand.grid(nrounds = c(50, 100,200), # max number of trees to build 
                       max_depth = c(15, 20, 30 ),
                       colsample_bytree = seq(0.7, 0.9, length.out = 2),
                       eta = c(0.01, 0.001), # learning rate
                       gamma=c(0.1, 0.01),
                       min_child_weight = c(1),
                       subsample = seq(0.5, 0.9, length.out = 5) # default values - unchanged 
)



set.seed(0) 
xgb_model_hyper2 = train( target ~., data = 
                           pca_50_train,
                         method = "xgbTree",
                         maximize = FALSE,
                         metric = 'rmsle',
                         tuneGrid = xgbGrid,
                         trControl = xgb_trcontrol
)

xgb_model_hyper2$bestTune
yhyper2.xgboost = predict(xgb_model_hyper2, newdata = pca_50_val)
(nrmse_hyper2_xgboost = hydroGOF::nrmse(yhyper2.xgboost , pca_50_val$target))
(rmsle_hyper2_xgboost = ModelMetrics::rmsle(abs(pca_50_val$target), abs(yhyper2.xgboost )))

range(xgb_model_hyper2$results$rmsle, na.rm = TRUE)
hist(xgb_model_hyper2$results$rmsle)




##### D. D e e p  L e a r n i n g  N N  ######
















