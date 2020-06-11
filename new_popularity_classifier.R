install.packages("e1071")
install.packages("lattice")
install.packages("caret")
install.packages("pROC")
install.packages("randomForest")
install.packages('caTools')
install.packages("car")
install.packages("rmarkdown")
library(car)
library(caTools)
library(randomForest)
library(pROC)
library(e1071)
library(lattice)
library(caret)
library(Boruta)
library(ggplot2)
library(gam)
options(scipen=999)

news_data <- read.csv("/Users/anuradha/Desktop/Spring/DS for BI/OnlineNewsPopularity/OnlineNewsPopularity.csv")
attach(news_data)
nrow(news_data)

drop_col =c("url","timedelta","is_weekend")
news_data <- news_data[,-match(drop_col,names(news_data))]
news_data <- news_data[!(news_data$n_tokens_content==0),]
nrow(news_data)

hist(news_data$shares,xlab = "Number of shares", main="Histogram of number of shares",border="blue", col="yellow")
hist(log(news_data$shares),xlab = "Log(Number of shares)", main="Histogram of Log Transformation",border="blue", col="yellow",breaks=50)
boxplot(news_data$shares,xlab = "Number of shares", main="Boxplot of number of shares",border="blue", col="yellow")
boxplot(log(news_data$shares),xlab = "Log(Number of shares)", main="Histogram of Log Transformation",border="blue", col="yellow",breaks=50)

#transform y
news_data$logshares <- log(news_data$shares)

#check publishing day vs shares
news_data$publishday <- rep("Sunday", nrow(news_data))
news_data$publishday[news_data$weekday_is_monday==1] <- "Monday"
news_data$publishday[news_data$weekday_is_tuesday==1] <- "Tuesday"
news_data$publishday[news_data$weekday_is_wednesday==1] <- "Wednesday"
news_data$publishday[news_data$weekday_is_thursday==1] <- "Thursday"
news_data$publishday[news_data$weekday_is_friday==1] <- "Friday"
news_data$publishday[news_data$weekday_is_saturday==1] <- "Saturday"
plot1 <- ggplot(data=news_data, aes(as.factor(publishday), news_data$logshares))
plot1+ geom_boxplot()
news_data <- subset(news_data, select= -c(publishday))
names(news_data)

#check type of news vs shares
news_data$type <- rep("Lifestyle", nrow(news_data))
news_data$type[news_data$data_channel_is_bus==1] <- "Business"
news_data$type[news_data$data_channel_is_entertainment==1] <- "Entertainment"
news_data$type[news_data$data_channel_is_socmed==1] <- "Social Media"
news_data$type[news_data$data_channel_is_tech==1] <- "Technology"
news_data$type[news_data$data_channel_is_world==1] <- "World"
plot2 <- ggplot(data=news_data, aes(as.factor(type), news_data$logshares))
plot2+ geom_boxplot()
news_data <- subset(news_data, select= -c(type))
names(news_data)

popularity <- ifelse(news_data$shares >= 1400, 1, 0)
popularity <- as.factor(popularity)
news_data$popularity <- popularity 
head(news_data)

for( i in 1:(length(news_data))){cat(names(news_data[i]),"->", skewness(news_data[,i]),"\n")}
news_data <- subset(news_data, select= -c(logshares))
names(news_data)

exclude_cols= c(12:17, 30:36, 58:59)
scaled_dataset <- scale(news_data[, -exclude_cols], center=T, scale=T)
scaled_dataset
temp_dataset<- data.frame(news_data[, exclude_cols], scaled_dataset)
head(temp_dataset)

#Feature Selection
set.seed(12345)
Boruta.news_train<- Boruta(popularity~., data=temp_dataset,  doTrace = 2, ntree = 100, maxRuns = 30)
print(Boruta.news_train)

features <- names(temp_dataset)[(which(Boruta.news_train$finalDecision == "Confirmed"))]  
dataset <- temp_dataset[, c("popularity", features)]
names(dataset)
plot(Boruta.news_train, cex.axis=.7, las=2, xlab="", main="Variable Importance",ylim=c(0,15))

set.seed(1000)
sample_size <- floor(0.7 * nrow(dataset))
train <- sample(seq_len(nrow(dataset)), size = sample_size)
train_set <- dataset[train, ]
test_set <- dataset[-train, ]
dim(train_set)

#logistic Accuracy : 0.6542    
log_fit = glm(popularity~.-shares, data=train_set,binomial)
summary(log_fit)
#plot(log_fit)

log_prediction <- predict(log_fit, test_set, type="response") 
accuracy<-ifelse(log_prediction>=0.5,1,0)
table(test_set$popularity)
confusionMatrix(factor(accuracy),factor(test_set$popularity))
plot(roc(test_set$popularity, log_prediction, direction="<"),
     col="red", lwd=3, main="ROC plot for Logistic")
log_roc=roc(test_set$popularity, log_prediction, direction="<")

#random forest Accuracy : 0.6637       
rf_fit = randomForest(popularity~.-shares, data=train_set, importance=TRUE)
summary(rf_fit)
#plot(rf_fit)

rf_prediction <- predict(rf_fit, test_set, type="class") 
#rf_accuracy<-ifelse(rf_prediction>=0.5,1,0)
table(test_set$popularity)
table(rf_prediction)
confusionMatrix(factor(rf_prediction),factor(test_set$popularity))
plot(roc(test_set$popularity, as.numeric(rf_prediction), direction="<"),col="red", lwd=3, main="ROC plot for Random Forest")
rf_roc=roc(test_set$popularity, as.numeric(rf_prediction), direction="<")

#Accuracy : 0.6529      
svm_fit= svm(popularity~.-shares,data=train_set, type = 'C-classification', kernel = 'polynomial')
summary(svm_fit)
svm_prediction<-predict(svm_fit,test_set)
confusionMatrix(factor(svm_prediction),factor(test_set$popularity))
plot(roc(test_set$popularity, as.numeric(svm_prediction), direction="<"),col="red", lwd=3, main="ROC plot for Support Vector Machine")
svm_roc=roc(test_set$popularity, as.numeric(svm_prediction), direction="<")

#all rocs
plot(log_roc,col= 2, main="ROC curves comparing classification performance")
legend(-0.2, 1.0, c('logistic', 'random forest', 'svm'), 2:4)
plot(rf_roc, col=3, add=TRUE)
plot(svm_roc, col=4, add=TRUE)

# train and testset for all predictors
names(temp_dataset)
news_data<-temp_dataset
sample_size <- floor(0.7 * nrow(news_data))
train <- sample(seq_len(nrow(news_data)), size = sample_size)
trainset <- news_data[train, ]
testset <- news_data[-train, ]

dim(trainset)
dim(testset)
names(trainset)

#glm + stepwise Accuracy - 0.6226 
s_fit= glm(popularity~.-shares,data=trainset, binomial)
step_fit=step(s_fit)
summary(step_fit)
yhat = predict(step_fit, testset)
accuracy<-ifelse(yhat>=0.5,1,0)
confusionMatrix(factor(accuracy),factor(testset$popularity))
#aic 33792.71
s_fit$aic
#aic 33770.53
step_fit$aic
# remove variables that are not part of step
new_fit<- update(step_fit,.~.-data_channel_is_world-weekday_is_sunday-n_tokens_title-n_non_stop_unique_tokens-num_imgs-num_videos-self_reference_max_shares-LDA_04-
                   global_sentiment_polarity-global_rate_negative_words-min_positive_polarity-max_positive_polarity-avg_negative_polarity-min_negative_polarity-
                   max_negative_polarity-abs_title_sentiment_polarity)

round(coef(new_fit),4)
step_prediction <- predict(new_fit, testset, type="response") 
step_accuracy<-ifelse(step_prediction>=0.5,1,0)
table(test_set$popularity)
table(step_accuracy)
#Accuracy 0.6534 
confusionMatrix(factor(step_accuracy),factor(testset$popularity))
vif(new_fit)
final_fit<- update(new_fit, .~.-kw_max_min-kw_avg_min-kw_max_avg-kw_avg_avg-rate_positive_words-rate_negative_words)
summary(final_fit)
plot(final_fit)
#aic=34247.39
final_fit$aic
step_prediction <- predict(final_fit, testset, type="response") 
step_accuracy<-ifelse(step_prediction>=0.5,1,0)
table(testset$popularity)
table(step_accuracy)
# Accurcay 0.6478      
confusionMatrix(factor(step_accuracy),factor(testset$popularity))

#ridge Accuracy : 0.6173       
library(glmnet)
x <- model.matrix(popularity~.-shares, trainset)[,-1]
y<- trainset$popularity
ridge_fit=glmnet(x, y, family = "binomial", alpha = 0)
plot(ridge_fit, xvar="lambda")
cv_fit = cv.glmnet(x, y, family = "binomial", alpha=0) 
cv_fit$lambda.min      
abline(v=log(cv_fit$lambda.min))
plot(cv_fit)
x_test <- model.matrix(popularity~.-shares, testset)[,-1]
yhat_ridge = predict(ridge_fit, s=cv_fit$lambda.1se, newx=x_test)
ridge_accuracy<-ifelse(yhat_ridge>=0.5,1,0)
confusionMatrix(factor(ridge_accuracy),factor(testset$popularity))

#lasso Accuracy : 0.6228
lasso_fit=glmnet(x, y, family = "binomial", alpha = 1)
summary(lasso_fit)
plot(lasso_fit, xvar="lambda")
cv_fit = cv.glmnet(x, y, family = "binomial", alpha=1) 
cv_fit$lambda.min      
abline(v=log(cv_fit$lambda.min))
plot(cv_fit)
x_test <- model.matrix(popularity~.-shares, testset)[,-1]
yhat_lasso = predict(lasso_fit, s=cv_fit$lambda.1se, newx=x_test)
lasso_accuracy<-ifelse(yhat_lasso>=0.5,1,0)
confusionMatrix(factor(lasso_accuracy),factor(testset$popularity))
summary(lasso_fit)

#gam
library(gam)
gamfit1 = gam(shares~data_channel_is_tech+
              weekday_is_thursday+
              s(n_non_stop_unique_tokens)+
                s(average_token_length)+
                s(kw_min_max)+
                s(kw_avg_avg)+
                s(LDA_01)+
                s(global_sentiment_polarity)+
                s(avg_positive_polarity)+
                s(max_negative_polarity)+
                data_channel_is_lifestyle+
                data_channel_is_world+
                weekday_is_friday+
                s(n_tokens_title)+
                s(num_hrefs)+
                s(num_keywords)+
                s(kw_max_max)+
                s(self_reference_min_shares)+
                s(LDA_02)+
                s(global_rate_positive_words)+
                s(min_positive_polarity)+
                s(title_subjectivity)+
                data_channel_is_entertainment+
                weekday_is_monday+
                weekday_is_saturday+
                s(n_tokens_content)+
                s(num_self_hrefs)+
                s(kw_min_min)+
                s(kw_avg_max)+
                s(self_reference_max_shares)+
                s(LDA_03)+
                s(global_rate_negative_words)+
                s(max_positive_polarity)+
                s(title_sentiment_polarity)+
                data_channel_is_bus+
                weekday_is_tuesday+
                weekday_is_sunday+
                s(n_unique_tokens)+
                s(num_imgs)+
                s(kw_max_min)+
                s(kw_min_avg)+
                s(self_reference_avg_sharess)+
                s(LDA_04)+
                s(rate_positive_words)+
                s(avg_negative_polarity)+
                s(abs_title_subjectivity)+
                data_channel_is_socmed+
              weekday_is_wednesday+
                n_non_stop_words+
                s(num_videos)+
                s(kw_avg_min)+
                s(kw_max_avg)+
                s(LDA_00)+
                s(global_subjectivity)+
                s(rate_negative_words)+
                s(min_negative_polarity)+
                s(abs_title_sentiment_polarity), data=trainset)
summary(gamfit1)
par(mfrow=c(2,4))
plot(gamfit1, se=T,ask=T)

#corelation matrices
correlations<- cor(as.matrix(news_data[, c(1:6,39:43)]))
correlations

data_channel_cor<- cor(as.matrix(news_data[, c(1:6)]))
data_channel_cor

weekday_cor<- cor(as.matrix(news_data[, c(7:13)]))
weekday_cor

kw_cor<- cor(as.matrix(news_data[, c(27:35)]))
kw_cor

lda_cor<- cor(as.matrix(news_data[, c(39:43)]))
lda_cor
