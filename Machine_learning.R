

#Classification And REgression Training
library(caret)
library(tidyverse)
library(AppliedPredictiveModeling)
library(pls) 
library(elasticnet)
library(broom) 
library(glmnet)
library(MASS)
library(ISLR)
library(PerformanceAnalytics)
library(Matrix) 
library(kernlab) 
library(e1071)
library(rpart)
library(pgmm) 
library(dslabs)
library(rpart.plot) 
library(partykit)
library(ipred) 
library(randomForest)
library(gbm)
library(nnet)
library(neuralnet)
library(GGally)
library(NeuralNetTools)
library(FNN)
library("readxl")
library(SHAPforxgboost)
library(xgboost)
library(data.table)
library(ggplot2)
library(hrbrthemes)
library(ggraph)
library(igraph)
library(tidyverse)
library(viridis)
library(corrgram)
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(ggridges)
library(h2o)

# Not all of these packages need to be installed. Caret package can be installed only for cross validation.


## Data

df <- read_xlsx('C:/Users/berka/Desktop/masaüstü_2023/freelance/ysa/data28.xlsx')






# Nonlinear Regression Model (KNN) with K Nearest Neighbor Algorithm


# First of all, we need to separate our data into train and test in order to test the accuracy of the prediction. Let's separate our data as 80% train and 20% test with the caret package. The set.seed command is used to avoid different results in each run.

summary(df)

set.seed(123)
train_indeks <- createDataPartition(df$Life_Satisfaction, 
                                    p = .7, 
                                    list = FALSE, 
                                    times = 1)
head(train_indeks)

train <- df[train_indeks, ]
test <- df[-train_indeks, ]
train_x <- train %>% dplyr::select(-Life_Satisfaction)
train_y <- train$Life_Satisfaction
test_x <- test %>% dplyr::select(-Life_Satisfaction)
test_y <- test$Life_Satisfaction

# a single dataset
training <- data.frame(train_x, Life_Satisfaction = train_y)


## Model 

# The following commands are used to set up the knn regression model on the train and test set. With the knn.reg command in the FNN package, a nonlinear regression model can be established by determining the k number of 3. The number of K neighbors is the parameter value that we need to determine here. With model tuning, the value with the least error of this parameter value will be selected.



library(FNN)
knn_fit <- knn.reg(train = train_x, 
                   test = test_x, 
                   y = train_y, 
                   k = 3)

names(knn_fit)



#"call""k""n""pred""residuals""PRESS""R2Pred" information can be examined optionally with the model we created. While k information shows us the number of neighbors we have determined, pred shows the estimated blood pressure values.


head(knn_fit$pred)




## Guess




knn_fit <- knn.reg(train = train_x, 
                   test = test_x, 
                   y = train_y, 
                   k = 3)

defaultSummary(data.frame(obs = test_y, pred = knn_fit$pred))


# When we compare the test_x blood pressure values estimated from the Train data with the real test_y data, the error metric values are found as RMSE=19, MAE=15 AND R squared explanatory value 3%, respectively. Although the error metric values were not excessively high, the R square value was quite low. It can be said that these errors will decrease when we will obtain the best k value with model tuning. 


## Model Tuning

# Let's try to reach the optimum k value by testing the model up to 35 k numbers. For this, let's write k numbers to try from 1 to 35 in knn_grid. knn should be selected as the method. Then, by applying the 10-fold cSpiritual_Well_Beings validation with the trainControl function in the caret package, let's get the k parameter value with the lowest RMSE error metric by applying hyperparameter optimization on the train data. In order to obtain this k value, the train function in the caret package can be used in the same way.


library(caret)
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)

knn_grid <- data.frame(k = 1:30)

knn_tune <- caret::train(train_x, train_y,
                         method = "knn",
                         trControl = ctrl,
                         tuneGrid = knn_grid,
                         preProc = c("center", "scale"))

ggplot(knn_tune)+
  ggtitle("Best Parameter Value for KNN") +
  xlab("Neighbors") + geom_line(color = "blue", size=1)
ggsave(file="C:/Users/berka/Desktop/knn.png", device='tiff', dpi=700)

# We can examine the change of RMSE values in the estimation made with the 10-fold cSpiritual_Well_Beings validation against the k value above. It can be seen that the k value with the lowest RMSE value is the number of 18 neighbors. With knn_tune, we can look at the RMSE R Squared and MAE values of these 35 k numbers. 


knn_tune


#The final value used for the model was k = 18 bilgisi ile k=18 komşuluk sayısı bize en optimal tahmin sonucunu vermektedir. Sırasıyla RMSE, Rsquared, MAE değerlerini  16.54460  0.17530278  13.23780 olarak elde ederiz.

# You can look inside the finalModel to see the exact best k parameter value.


knn_tune$finalModel


# The RMSE (16.54), Rsquared (0.17), MAE (13.23) results obtained above were valid for the train data. When we compare the predicted values with the test data, the RMSE, Rsquared, MAE values are as follows.


defaultSummary(data.frame(obs = test_y,
                          pred = predict(knn_tune, test_x)))


# In this case, we say that tuned k values give better results than trial and error.


# Support Vector Regression (SVR)


## Model



library(e1071)
library(kernlab)
library(purrr)
library(ggplot2)
library(PerformanceAnalytics)



svm_fit <- svm(train_x, train_y)
svm_fit




names(svm_fit)




svm_fit$epsilon





## Model Tuning


## Cross Validation


set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
svm_tune <- caret::train(train_x, train_y,
                         method = "svmRadial",
                         trControl = ctrl,
                         tuneGrid = data.frame(.C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5,2,2.5,3),
                                               .sigma = .05),
                         tuneLength = 7,
                         preProc = c("center", "scale"))

plot(svm_tune, main = "Best Parameter Value for SVM", xlab= "Cost")


ggplot(svm_tune)+
  ggtitle("Best Parameter Value for SVM") +
  xlab("Cost") + geom_line(color = "blue", size=1)
ggsave(file="C:/Users/berka/Desktop/svm.png", device='tiff', dpi=700)


svm_tune$finalModel





defaultSummary(data.frame(obs = test_y,
                          pred = predict(svm_tune, test_x)))




# NNA





## Model Tuning



library(RSNNS)

set.seed(123)
#Multi-Layer Perceptron
ctrl <- trainControl(method = "cv", number = 10)

ysa_grid <- expand.grid(
  decay = c(0.001,0.01,0.05, 0.1, 0.2, 0.4),
  size =  (1:10), bag = c(T,F))

ysa_tune <- caret::train(train_x, train_y,
                         method = "avNNet",
                         trControl = ctrl,
                         tuneGrid = ysa_grid,
                         preProc = c("center", "scale"),
                         linout = TRUE, maxit = 100)

plot(ysa_tune, main = "Best Parameter Value for ANN", xlab= "Hidden Units")

ggplot(ysa_tune)+
  ggtitle("Best Parameter Value for ANN") +
  xlab("Neighbors")
ggsave(file="C:/Users/berka/Desktop/ysa.png", device='tiff', dpi=700)



ysa_tune$bestTune
defaultSummary(data.frame(obs = test_y,
                          pred = predict(ysa_tune, test_x)))



# Random Forests Regresyon



## Model 


rf_fit <- randomForest(train_x, train_y, importance = TRUE)

importance(rf_fit)
varImpPlot(rf_fit)


rf_fit


## 


predict(rf_fit, test_x )

plot(predict(rf_fit, test_x), test_y,
     xlab = "Tahmin Edilen", ylab = "Gercek",
     main = "Tahmin Edilen vs Gercek: Random Forest",
     col = "dodgerblue", pch = 20)

grid()
abline(0, 1, col = "darkorange", lwd = 2)


defaultSummary(data.frame(obs = test_y,
                          pred = predict(rf_fit, test_x)))




## Model Tunning



set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)


ncol(train_x)/3

tune_grid <- expand.grid(mtry = c(1,2,3,4,5,8,10,12,15,17,20))
rf_tune <- caret::train( train_x, train_y,
                         method = "rf",
                         tuneGrid = tune_grid,
                         trControl = ctrl
                         
)

rf_tune
plot(rf_tune, main = "Best Parameter Value for RF", xlab= "Randomly Selected Predictors")

ggplot(rf_tune)+
  ggtitle("Best Parameter Value for RF") +
  xlab("Randomly Selected Predictors") + geom_line(color = "blue", size=1)
ggsave(file="C:/Users/berka/Desktop/rf.png", device='tiff', dpi=700)

rf_tune$results %>% filter(mtry == as.numeric(rf_tune$bestTune))

defaultSummary(data.frame(obs = test_y, 
                          pred = predict(rf_tune, test_x)))




defaultSummary(data.frame(obs = test_y,
                          pred = predict(rf_tune, test_x)))



#XGBoost
## Model 


library(xgboost)


Model


xgboost_fit <-xgboost(data = as.matrix(train_x),
                      label = train_y, 
                      booster = "gblinear",
                      max.depth = 2,
                      eta = 1,
                      nthread = 2, 
                      nrounds = 1000)


dtrain <- xgb.DMatrix(data = as.matrix(train_x), label = train_y)
dtest <- xgb.DMatrix(data = as.matrix(test_x), label = test_y)

xgboost_fit <-xgboost(data = dtrain, 
                      booster = "gblinear",
                      max.depth = 2,
                      eta = 1,
                      nthread = 2, 
                      nrounds = 3)

xgboost_fit


class(dtrain)


imp_matris <- xgb.importance(model = xgboost_fit)
imp_matris

xgb.plot.importance(imp_matris)








watchlist <- list(train = dtrain, test = dtest)

xgb_fit <- xgb.train(data = dtrain, 
                     booster = "gblinear",
                     max.depth = 4,
                     eta = 0.1, 
                     nthread = 2,
                     nrounds = 100,
                     watchlist = watchlist)





## Guess


predict(xgb_fit, as.matrix(test_x))

plot(predict(xgb_fit, as.matrix(test_x)), test_y,
     xlab = "Tahmin Edilen", ylab = "Gercek",
     main = "Tahmin Edilen vs Gercek: XGBoost",
     col = "dodgerblue", pch = 20)
grid()
abline(0, 1, col = "darkorange", lwd = 2)


defaultSummary(data.frame(obs = test_y, 
                          pred = predict(xgb_fit, as.matrix(test_x))))




## Model Tuning

set.seed(123)

ctrl <- trainControl(method = "cv", number = 10)

xgb_grid <- expand.grid(
  nrounds = 1000,
  lambda = c(0,0.01, 0.05, 0.1, 0.5, 1, 1.5,2),
  alpha = c(0,0.1, 0.5, 1, 2, 3),
  eta = c(1)
  
)


xgb_tune_fit <- caret::train(
  x = data.matrix(train_x),
  y = train_y,
  trControl = ctrl,
  tuneGrid = xgb_grid,
  method = "xgbLinear"
)

defaultSummary(data.frame(obs = test_y, 
                          pred = predict(xgb_tune_fit, as.matrix(test_x))))






plot(xgb_tune_fit, main = "Best Parameter Value for XGBoost", xlab="L1 Regularization (eta) = 1")
ggplot(xgb_tune_fit)+
  ggtitle("Best Parameter Value for XGBoost") +
  xlab("L1 Regularization (eta) = 1")
ggsave(file="C:/Users/berka/Desktop/xg.png", device='tiff', dpi=700)




xgb_tune_fit$bestTune



# Regression
## Model Kurma


## Model Tuning


set.seed(123)
ctrl <- trainControl(method = "cv",repeats = 10,verboseIter =TRUE)


eGrid <- expand.grid(.alpha = (0:10) * 0.1, 
                     .lambda = (0:10) * 0.1)


reg_tune <- caret::train(data.matrix(train_x), train_y,
                         method = "glmnet",
                         tuneGrid = eGrid,
                         trControl = ctrl)


reg_tune



plot(reg_tune, main = "Best Parameter Value for Regression")
ggplot(reg_tune)+
  ggtitle("Best Parameter Value for Regression") 
ggsave(file="C:/Users/berka/Desktop/reg.png", device='tiff', dpi=700)

reg_tune$bestTune
reg_tune$finalModel

plot(reg_tune$finalModel)


plot(predict(reg_tune), train$Life_Satisfaction, xlab = "Tahmin",ylab = "Gercek")




defaultSummary(data.frame(obs = train$Life_Satisfaction,
                          pred = predict(reg_tune)))




# Cart
## Model Kurma


cart_tree <- rpart(Life_Satisfaction ~ ., data = df)

names(cart_tree)
cart_tree$variable.importance

plot(cart_tree, margin = 0.1)
text(cart_tree, cex = 0.5)
prp(cart_tree, type = 4)
rpart.plot(cart_tree)
plotcp(cart_tree)







## Model Tuning



set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)

tune_grid <- expand.grid(.maxdepth=seq(0,10,0.5))

cart_tune <- caret::train(Life_Satisfaction~.,
                          method = "rpart2",
                          trControl = ctrl,
                          tuneGrid = tune_grid,
                          preProc = c("center", "scale"), 
                          data = train, metric = 'RMSE')

cart_tune

plot(cart_tune, main = "Best Parameter Value for CART")
ggplot(cart_tune)+
  ggtitle("Best Parameter Value for CART") +
  geom_line(color = "blue", size=1)
ggsave(file="C:/Users/berka/Desktop/cart.png", device='tiff', dpi=700)

cart_tune$bestTune
cart_tune$finalModel
rpart.plot(cart_tune$finalModel)

plot(predict(cart_tune), train$Life_Satisfaction, xlab = "Tahmin",ylab = "Gercek")
abline(0,1)



defaultSummary(data.frame(obs = train$Life_Satisfaction,
                          pred = predict(cart_tune)))



# Sonuc

KNN<-defaultSummary(data.frame(obs = test_y,
                               pred = predict(knn_tune, test_x)))
KNN
knn_tune$finalModel



SVM<-defaultSummary(data.frame(obs = test_y,
                               pred = predict(svm_tune, test_x)))
SVM
svm_tune$finalModel



ANN<-defaultSummary(data.frame(obs = test_y,
                               pred = predict(ysa_tune, test_x)))
ANN
ysa_tune$finalModel
ysa_tune$bestTune


RF<-defaultSummary(data.frame(obs = test_y,
                              pred = predict(rf_tune, test_x)))
RF
rf_tune$finalModel



XGBoost<-defaultSummary(data.frame(obs = test_y,
                                   pred = predict(xgb_tune_fit, newdata=test_x)))
XGBoost
xgb_tune_fit$finalModel
xgb_tune_fit$bestTune


CART<-defaultSummary(data.frame(obs = test_y,
                                pred = predict(cart_tune, newdata= test_x)))
CART
cart_tune$finalModel
cart_tune$bestTune


REG<-defaultSummary(data.frame(obs = test_y,
                               pred = predict(reg_tune, test_x)))
REG
reg_tune$bestTune

KNN
SVM
ANN
RF
XGBoost
CART
REG








toplam<- cbind(KNN,CART,SVM,ANN,XGBoost,RF,REG)
toplam <- rbind(toplam[-2, replace = T])
Output <- data.frame( Comparison = c("RMSE", "MAE"))
toplam<- cbind(Output,toplam)
toplam<-as.data.frame(toplam)
toplam <- toplam %>% gather(Method, Values, KNN:REG)



ggplot(toplam,                 
       aes(x = Method,
           y = Values,
           fill = Comparison)) +
  geom_bar(stat = "identity",
           position = "dodge")+labs(fill = "Metrics")


ggplot(toplam, aes(x = Method, y = Values, 
                   color = Comparison, group = Comparison)) + 
  geom_line() + geom_point()+labs(fill = "Metrics")
ggsave(file="C:/Users/berka/Desktop/hist.png", device='tiff', dpi=700)


test_pred_grid<-predict(xgb_tune_fit, test_x)
tahmin<-cbind(test_pred_grid,
              test_y)
tahmin<-as.data.frame(tahmin)
colnames(tahmin) <- c('Prediction','Observation')
tahmin <- tahmin %>% gather(tahmin1, deger, Prediction:Observation)



plot(test_y,type='l',col='dodgerblue',main = 'Estimated vs Observed Values',
     xlab = 'Number of Observations',ylab = 'Value Range',pch=19,lwd=2,cex=1.5, ylim = c(0.5,7))
lines(test_pred_grid,type = 'l',col='red',pch=19,lwd=2,cex=1.5)

legend("bottom", inset=.02, title="Data Status",
       c("Observed","Estimated"), fill = c("dodgerblue","red"), horiz=TRUE, cex=0.8)

