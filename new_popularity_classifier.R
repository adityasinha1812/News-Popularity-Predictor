install.packages("e1071")
install.packages("lattice")
install.packages("caret")
install.packages("pROC")
install.packages("randomForest")
install.packages('caTools')
install.packages("car")
library(car)
library(caTools)
library(randomForest)
library(pROC)
library(e1071)
library(lattice)
library(caret)
library(Boruta)
library(ggplot2)

news_data <- read.csv("/Users/anuradha/Desktop/Spring/DS for BI/OnlineNewsPopularity/OnlineNewsPopularity.csv")
attach(news_data)
nrow(news_data)

drop_col =c("url","timedelta","is_weekend")
news_data <- news_data[,-match(drop_col,names(news_data))]


options(scipen=999)
par("mar") 
par(mar=c(1,1,1,1))
par(mfrow=c(2,2))
for(i in 1:length(news_data)){boxplot(news_data[,i], xlab=names(news_data)[i] , main = paste("[" , i , "]" , "Histogram of", names(news_data)[i])  )}

par(mfrow=c(2,2))
for(i in 1:length(news_data)){hist(news_data[,i],xlab=names(news_data)[i] , main = paste("[" , i , "]" ,"Histogram of", names(news_data)[i])  )}

news_data <- news_data[!(news_data$n_tokens_content==0),]
news_data <- news_data[(news_data$n_non_stop_words<=1),]
news_data <- news_data[!(news_data$n_unique_tokens==701),]
nrow(news_data)

hist(news_data$shares,xlab = "Number of shares", main="Histogram of number of shares",border="blue", col="yellow")
hist(log(news_data$shares),xlab = "Log(Number of shares)", main="Histogram of Log Transformation",border="blue", col="yellow",breaks=50)
boxplot(news_data$shares,xlab = "Number of shares", main="Boxplot of number of shares",border="blue", col="yellow")
boxplot(log(news_data$shares),xlab = "Log(Number of shares)", main="Histogram of Log Transformation",border="blue", col="yellow",breaks=50)

#transform y
news_data$logshares <- log(news_data$shares)


#print outlier, criteria= 3 * IQR
#source : https://stackoverflow.com/questions/14207739/how-to-remove-outliers-in-boxplot-in-r
outliers<-boxplot(news_data$logshares, outline=FALSE, range=3)
print(outliers$out)

outliersstats <- boxplot(news_data$logshares)$stats
print(outliersstats)

min <- outliersstats[1,1]
lowerthreshold <- outliersstats[2,1]
median <- outliersstats[3,1]
upperthreshold <- outliersstats[4,1]
max <- outliersstats[5,1]

cat(min,lowerthreshold,median,upperthreshold,max)

iqr<- upperthreshold- lowerthreshold
lower <- lowerthreshold- 3* iqr
higher <- upperthreshold + 3* iqr
cat (iqr,lower, higher)

#remove outliers
news_data <- news_data[!(news_data$logshares >= higher | news_data$logshares <= lower),]

boxplot(log(news_data$shares),xlab = "Log(Number of shares)", main="Histogram of Log Transformation",border="blue", col="yellow",breaks=50)

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

#two categories, popular vs un-popular
popularity<- cut(news_data$logshares,c(lower,median,higher),labels=c(0,1) )
#lapply(news_data,function(x) { length(which(is.na(x)))})

#If skewness value lies above +1 or below -1, data is highly skewed. If it lies between +0.5 to -0.5, it is moderately skewed. If the value is 0, then the data is symmetric
for( i in 1:(length(news_data))){cat(names(news_data[i]),"->", skewness(news_data[,i]),"\n")}
news_data <- subset(news_data, select= -c(logshares))
dataset<- news_data
news_data$popularity <- popularity 
names(news_data)

#https://medium.com/@TheDataGyan/day-8-data-transformation-skewness-normalization-and-much-more-4c144d370e55
exclude_cols= c(12:17, 30:36, 58)
scaled_dataset <- scale(dataset[, -exclude_cols], center=T, scale=T)
scaled_dataset
temp_dataset<- data.frame(popularity, dataset[, exclude_cols], scaled_dataset)
names(temp_dataset)

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


#logistic
log_fit = glm(popularity~., data=train_set,binomial)
summary(log_fit)
plot(log_fit)

log_prediction <- predict(log_fit, test_set, type="response") 
accuracy<-ifelse(log_prediction>=0.5,1,0)
table(test_set$popularity)

#source:https://stackoverflow.com/questions/30002013/error-in-confusion-matrix-the-data-and-reference-factors-must-have-the-same-nu
confusionMatrix(factor(accuracy),factor(test_set$popularity))

#source:https://blog.revolutionanalytics.com/2016/08/roc-curves-in-two-lines-of-code.html
plot(roc(test_set$popularity, log_prediction, direction="<"),
     col="red", lwd=3, main="ROC plot for Logistic")
log_roc=roc(test_set$popularity, log_prediction, direction="<")

#random forest
rf_fit = randomForest(popularity~., data=train_set, importance=TRUE)
summary(rf_fit)
plot(rf_fit)

rf_prediction <- predict(rf_fit, test_set, type="class") 
table(test_set$popularity)
table(rf_prediction)
confusionMatrix(factor(rf_prediction),factor(test_set$popularity))
plot(roc(test_set$popularity, as.numeric(rf_prediction), direction="<"),
     col="red", lwd=3, main="ROC plot for Random Forest")
rf_roc=roc(test_set$popularity, as.numeric(rf_prediction), direction="<")

#SVM
svm_fit= svm(popularity~.,data=train_set)
svm_prediction<-predict(svm_fit,test_set)
confusionMatrix(factor(svm_prediction),factor(test_set$popularity))
plot(roc(test_set$popularity, as.numeric(svm_prediction), direction="<"),
     col="red", lwd=3, main="ROC plot for Support Vector Machine")
svm_roc=roc(test_set$popularity, as.numeric(svm_prediction), direction="<")

#all rocs
plot(log_roc,col= 2, main="ROC curves comparing classification performance")
legend(-0.2, 1.0, c('logistic', 'random forest', 'svm'), 2:4)
plot(rf_roc, col=3, add=TRUE)
plot(svm_roc, col=4, add=TRUE)


#stepwise 
names(news_data)
set.seed(1000)
sample_size <- floor(0.7 * nrow(news_data))
train <- sample(seq_len(nrow(news_data)), size = sample_size)
trainset <- news_data[train, ]
testset <- news_data[-train, ]

dim(trainset)
dim(testset)

#glm + step 
s_fit= glm(popularity~.-shares,data=trainset, binomial)
step_fit=step(s_fit)
summary(step_fit)
plot(step_fit)
names(news_data)

#aic
s_fit$aic
step_fit$aic

new_fit <- update(s_fit,.~.-n_tokens_title-n_unique_tokens-n_non_stop_words-n_non_stop_unique_tokens-num_imgs-num_videos-data_channel_is_world-self_reference_avg_sharess-
weekday_is_sunday-LDA_04-global_rate_negative_words-rate_negative_words-avg_positive_polarity-max_positive_polarity-avg_negative_polarity-
min_negative_polarity-max_negative_polarity-abs_title_sentiment_polarity )

round(coef(new_fit),4)

step_prediction <- predict(new_fit, testset, type="response") 
step_accuracy<-ifelse(step_prediction>=0.5,1,0)

table(test_set$popularity)
table(step_accuracy)

#0.6456  
confusionMatrix(factor(step_accuracy),factor(testset$popularity))

#check multicolinearity
vif(new_fit)

#source:http://www.sthda.com/english/articles/39-regression-model-diagnostics/160-multicollinearity-essentials-and-vif-in-r/
#remove variables with vif over 5:- kw_max_min,kw_avg_min,kw_max_avg,kw_avg_avg

final_fit<- update(new_fit, .~.-kw_max_min-kw_avg_min-kw_max_avg-kw_avg_avg)
summary(final_fit)
plot(final_fit)
final_fit$aic
step_prediction <- predict(final_fit, testset, type="response") 
step_accuracy<-ifelse(step_prediction>=0.5,1,0)

table(testset$popularity)
table(step_accuracy)

#0.6353   
confusionMatrix(factor(step_accuracy),factor(testset$popularity))


summary(final_fit)
coef(final_fit)





#https://www.guru99.com/r-generalized-linear-model.html
library(ROCR)
ROCRpred <- prediction(step_prediction, testset$popularity)
ROCRperf <- performance(ROCRpred, 'tpr', 'fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2, 1.7))





#step uses add() and drop1() already so try anova()


