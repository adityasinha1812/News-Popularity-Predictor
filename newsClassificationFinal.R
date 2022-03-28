# importing libraries
library(car)
library(caTools)
library(randomForest)
library(glmnet)
library(pROC)
library(e1071)
library(lattice)
library(caret)
library(Boruta)
library(ggplot2)
library(gam)

# penalty to be applied when deciding to print numeric values in fixed or exponential notation.
options(scipen = 999)

# importing data
news = read.csv("C:/Users/adity/Downloads/Courses/Spring 2020/MSIT 423 - Data Science/Online News Popularity/OnlineNewsPopularity.csv")

# number of rows check - 39644
nrow(news)

# removing data for which there is no content
news <- news[!(news$n_tokens_content == 0), ]

# number of rows check - 38463
nrow(news)

# looking at the dependent variable (shares)
summary(news$shares)
boxplot(news$shares) # can observe outliers
qqnorm(news$shares) # not normal (very right skewed)
qqline(news$shares)
hist(news$shares, prob = T) # very right skewed
lines(density(news$shares), col = "blue")

# looking at the log of dependent variable (shares)
summary(log(news$shares))
boxplot(log(news$shares)) # much more balanced
qqnorm(log(news$shares)) # straighter line (indicates normal)
qqline(log(news$shares))
hist(log(news$shares), prob = T) # normal distribution
lines(density(log(news$shares)), col = "blue")

# lets log the share variable
news$shares = log(news$shares)

# check publishing day vs shares
news$publishday = rep("Sunday", nrow(news))
news$publishday[news$weekday_is_monday == 1] = "Monday"
news$publishday[news$weekday_is_tuesday == 1] = "Tuesday"
news$publishday[news$weekday_is_wednesday == 1] = "Wednesday"
news$publishday[news$weekday_is_thursday == 1] = "Thursday"
news$publishday[news$weekday_is_friday == 1] = "Friday"
news$publishday[news$weekday_is_saturday == 1] = "Saturday"
plot_day = ggplot(data = news, aes(as.factor(publishday), shares))
plot_day + geom_boxplot()
news = subset(news, select = -c(publishday))
names(news_data)

# check channel type vs shares
news$type = rep("Lifestyle", nrow(news))
news$type[news$data_channel_is_bus == 1] = "Business"
news$type[news$data_channel_is_entertainment == 1] = "Entertainment"
news$type[news$data_channel_is_socmed == 1] = "Social Media"
news$type[news$data_channel_is_tech == 1] = "Technology"
news$type[news$data_channel_is_world == 1] = "World"
plot_type = ggplot(data = news, aes(as.factor(type), shares))
plot_type + geom_boxplot()
news = subset(news, select = -c(type))
names(news_data)

# removing url, timedelta (non predictive) and is_weekend (repetitive) at the outset
news <- subset(news, select = -c(url, timedelta, weekday_is_sunday, weekday_is_monday,
                                 weekday_is_tuesday, weekday_is_wednesday, weekday_is_thursday,
                                 weekday_is_friday, weekday_is_saturday))

# creating popularity column based on median value
news$popularity = ifelse(news$shares >= 7.244, 1, 0)

par(mfrow = c(2,2))
plot(log(news$self_reference_min_shares), log(news$shares))
plot(log(news$kw_avg_avg), log(news$shares))
plot(news$is_weekend, log(news$shares))
plot(log(news$kw_max_avg), log(news$shares))
plot(log(news$self_reference_avg_sharess), log(news$shares))
plot(news$data_channel_is_socmed, log(news$shares))
plot(news$LDA_00, log(news$shares))
plot(news$LDA_02, log(news$shares))
plot(log(news$kw_min_avg + 1), log(news$shares))
plot(news$data_channel_is_entertainment, log(news$shares))

# removing shares column since we are working with popularity
news = subset(news, select = -c(shares))

# feature selection
set.seed(12345)
boruta1 = Boruta(popularity ~ ., data = news, doTrace = 2, ntree = 200, maxRuns = 20)
print(boruta1)
features1 = names(news)[(which(boruta1$finalDecision == "Confirmed"))]  
news1 = news[, c("popularity", features1)]
names(news1)
plot(boruta1, cex.axis = .7, las = 2, xlab = "", 
     main = "Variable Importance", ylim = c(0, 15))
attStats(boruta1)

# creating train and test set (70 / 30 split)
set.seed(12345)
train = sample(nrow(news1), as.integer(nrow(news1) * 0.75))
newsTrain = news1[train, ]
newsTest = news1[-train, ]
newsTrainAll = news[train, ]
newsTestAll = news[-train, ]

# method 0: logistic regression on all (65.8)
fit0 <- glm(popularity ~ ., data = newsTrainAll, binomial, maxit = 100)
alias(fit0)
summary(fit0)
plot(fit0)
pred0 = predict(fit0, newsTestAll, type = "response") 
acc0 = ifelse(pred0 >= 0.5, 1, 0)
confusionMatrix(factor(acc0), factor(newsTestAll$popularity))
plot(roc(newsTestAll$popularity, as.numeric(pred0), direction = "<"), main = "ROC - Logistic Regression")
roc0 = roc(newsTestAll$popularity, as.numeric(pred0), direction = "<")

# method 0.1: stepwise accuracy (63)
fit01 = step(fit0)
summary(fit01)
pred01 = predict(fit01, newsTestAll)
acc01 = ifelse(pred01 >= 0.5, 1, 0)
confusionMatrix(factor(acc01), factor(newsTestAll$popularity))
plot(roc(newsTestAll$popularity, as.numeric(pred01), direction = "<"), main = "ROC - Stepwise")
roc01 = roc(newsTestAll$popularity, as.numeric(pred01), direction = "<")

# taking only important variables
news2 = subset(news, select = c(popularity, 
                                data_channel_is_socmed,
                                data_channel_is_tech,
                                kw_max_avg,
                                kw_avg_avg,
                                is_weekend,
                                LDA_02,
                                LDA_00,
                                data_channel_is_entertainment,
                                num_hrefs,
                                kw_min_min,
                                global_subjectivity,
                                kw_min_avg,
                                num_keywords,
                                num_self_hrefs,
                                LDA_03,
                                LDA_01,
                                self_reference_avg_sharess,
                                n_tokens_content,
                                kw_avg_max,
                                n_non_stop_unique_tokens,
                                kw_avg_min,
                                min_positive_polarity,
                                kw_max_min,
                                abs_title_subjectivity,
                                global_rate_positive_words,
                                n_non_stop_words,
                                kw_min_max,
                                average_token_length,
                                title_subjectivity,
                                data_channel_is_bus,
                                self_reference_min_shares,
                                title_sentiment_polarity,
                                kw_max_max,
                                rate_positive_words,
                                min_negative_polarity,
                                global_rate_negative_words
))

# feature selection 2
set.seed(12345)
boruta2 = Boruta(popularity ~ ., data = news2, doTrace = 2, ntree = 100, maxRuns = 20)
print(boruta2)
features2 = names(news2)[(which(boruta2$finalDecision == "Confirmed"))]  
news3 = news2[, c(features2)]
plot(boruta2, cex.axis = .7, las = 2, xlab = "", 
     main = "Variable Importance", ylim = c(0, 15))
attStats(boruta2)
names(news3)

summary(news3)

newsTrainAll2 = news3[train, ]
newsTestAll2 = news3[-train, ]

# method 0: logistic regression on all (65.84)
fit02 <- glm(popularity ~ ., data = newsTrainAll2, binomial, maxit = 100)
alias(fit02)
summary(fit02)
plot(fit02)
pred02 = predict(fit02, newsTestAll2, type = "response") 
acc02 = ifelse(pred02 >= 0.5, 1, 0)
confusionMatrix(factor(acc02), factor(newsTestAll2$popularity))

# method 0.1: stepwise accuracy (63.01)
fit03 = step(fit02)
summary(fit03)
pred03 = predict(fit03, newsTestAll2)
acc03 = ifelse(pred03 >= 0.5, 1, 0)
confusionMatrix(factor(acc03), factor(newsTestAll2$popularity))
vif(fit03)

# method 1: logistic regression (65.89)
fit1 <- glm(popularity ~ ., data = newsTrain, binomial, maxit = 100)
summary(fit1)
plot(fit1)
pred1 = predict(fit1, newsTest, type = "response") 
acc1 = ifelse(pred1 >= 0.5, 1, 0)
confusionMatrix(factor(acc1), factor(newsTest$popularity))
plot(roc(newsTest$popularity, as.numeric(pred1), direction = "<"), main = "ROC - Logistic Regression")
roc1 = roc(newsTest$popularity, as.numeric(pred1), direction = "<")

# method 2: random forest accuracy (66.93)       
fit2 = randomForest(popularity ~ ., data = newsTrain, importance = TRUE)
summary(fit2)
plot(fit2)
pred2 = predict(fit2, newsTest, type = "class") 
acc2 = ifelse(pred2 >= 0.5, 1, 0)
confusionMatrix(factor(acc2), factor(newsTest$popularity))
plot(roc(newsTest$popularity, as.numeric(pred2), direction = "<"), main = "ROC - Random Forest")
roc2 = roc(newsTest$popularity, as.numeric(pred2), direction = "<")

# method 2_1: random forest accuracy (67.65)       
fit2_1 = randomForest(popularity ~ ., data = newsTrainAll2, importance = TRUE, ntrees = 100)
summary(fit2_1)
plot(fit2_1)
importance(fit2_1)
pred2_1 = predict(fit2_1, newsTestAll2, type = "class") 
acc2_1 = ifelse(pred2_1 >= 0.5, 1, 0)
confusionMatrix(factor(acc2_1), factor(newsTestAll2$popularity))
vif(fit2_1)

# method 3: svm accuracy (65.26)      
fit3 = svm(popularity ~ ., data = newsTrain, type = 'C-classification', kernel = 'polynomial')
summary(fit3)
pred3 = predict(fit3, newsTest) 
confusionMatrix(factor(pred3), factor(newsTest$popularity))
plot(roc(newsTest$popularity, as.numeric(pred3), direction = "<"), main = "ROC - SVM")
roc3 = roc(newsTest$popularity, as.numeric(pred3), direction = "<")

# plotting all rocs
plot(roc1, col = 2, main = "ROC curves comparing classification performance")
legend(-0.2, 1.0, c('Logistic', 'Random Forest', 'SVM'), 2:4)
plot(roc2, col = 3, add = TRUE)
plot(roc3, col = 4, add = TRUE)

# method 4: stepwise accuracy (62.98)
fit4 = step(fit1)
summary(fit4)
pred4 = predict(fit4, newsTest)
acc4 = ifelse(pred4 >= 0.5, 1, 0)
confusionMatrix(factor(acc4), factor(newsTest$popularity))
plot(roc(newsTest$popularity, as.numeric(pred4), direction = "<"), main = "ROC - Stepwise")
roc4 = roc(newsTest$popularity, as.numeric(pred4), direction = "<")

# method 5: ridge regression accuracy (62.71)
x = model.matrix(popularity ~ ., newsTrain)[, -1]
y = newsTrain$popularity
fit5 = glmnet(x, y, family = "binomial", alpha = 0)
plot(fit5, xvar = "lambda")
cvfit5 = cv.glmnet(x, y, family = "binomial", alpha = 0) 
cvfit5$lambda.min
abline(v = log(cvfit5$lambda.min))
plot(cvfit5)
xtest = model.matrix(popularity ~ ., newsTest)[, -1]
pred5 = predict(fit5, s = cvfit5$lambda.min, newx = xtest)
acc5 = ifelse(pred5 >= 0.5, 1, 0)
confusionMatrix(factor(acc5), factor(newsTest$popularity))

# method 6: lasso regression accuracy (63.11)
fit6 = glmnet(x, y, family = "binomial", alpha = 1)
summary(fit6)
plot(fit6, xvar = "lambda")
cvfit6 = cv.glmnet(x, y, family = "binomial", alpha = 1) 
cvfit6$lambda.min
abline(v = log(cvfit6$lambda.min))
plot(cvfit6)
pred6 = predict(fit6, s = cvfit6$lambda.min, newx = xtest)
acc6 = ifelse(pred6 >= 0.5, 1, 0)
confusionMatrix(factor(acc6), factor(newsTest$popularity))
summary(fit6)

# method 6.1: lasso regression accuracy (63.08)
x_all = model.matrix(popularity ~ ., newsTrainAll)[, -1]
y_all = newsTrainAll$popularity
xtestall = model.matrix(popularity ~ ., newsTestAll)[, -1] 
fit6_1 = glmnet(x_all, y_all, family = "binomial", alpha = 1)
summary(fit6_1)
plot(fit6_1, xvar = "lambda")
cvfit6_1 = cv.glmnet(x_all, y_all, family = "binomial", alpha = 1) 
cvfit6_1$lambda.min
abline(v = log(cvfit6_1$lambda.min))
plot(cvfit6_1)
pred6_1 = predict(fit6_1, s = cvfit6_1$lambda.min, newx = xtestall)
acc6_1 = ifelse(pred6_1 >= 0.5, 1, 0)
confusionMatrix(factor(acc6_1), factor(newsTestAll$popularity))
summary(fit6_1)
fit6_1$beta
as.matrix(coef(cvfit6_1, cvfit6_1$lambda.min))

summary(newsTrainAll2)
gamfit = gam(popularity ~ rate_positive_words + s(self_reference_min_shares) +      s(kw_avg_avg) + is_weekend + s(kw_max_avg) + s(self_reference_avg_sharess) +      data_channel_is_socmed + LDA_00 + s(LDA_02) + s(kw_min_avg) +      data_channel_is_entertainment + n_non_stop_words + LDA_01 +      data_channel_is_tech + s(kw_min_max) + s(n_tokens_content) +      s(num_hrefs) + LDA_03 + s(kw_avg_min) + s(n_non_stop_unique_tokens) +      kw_avg_max + s(kw_max_max) + s(num_self_hrefs) + s(global_rate_positive_words) +      s(global_subjectivity) + s(kw_min_min) + s(min_positive_polarity) +      s(num_keywords) + data_channel_is_bus + title_sentiment_polarity +s(title_subjectivity), data = newsTrainAll2, family = "binomial")


# accuracy 64.29
gamfit1 = step.Gam(gamfit, scope = list(
  'self_reference_min_shares'=~1+self_reference_min_shares+s(self_reference_min_shares),
  'kw_avg_avg'=~1+kw_avg_avg+s(kw_avg_avg),
  'is_weekend'=~1+is_weekend,
  'kw_max_avg'=~1+kw_max_avg+s(kw_max_avg),
  'self_reference_avg_sharess'=~1+self_reference_avg_sharess+s(self_reference_avg_sharess),
  'data_channel_is_socmed'=~1+data_channel_is_socmed,
  'LDA_00'=~1+LDA_00+s(LDA_00),
  'LDA_02'=~1+LDA_02+s(LDA_02),
  'kw_min_avg'=~1+kw_min_avg+s(kw_min_avg),
  'data_channel_is_entertainment'=~1+data_channel_is_entertainment,
  'n_non_stop_words'=~1+n_non_stop_words,
  'LDA_01'=~1+LDA_01+s(LDA_01),
  'data_channel_is_tech'=~1+data_channel_is_tech,
  'kw_min_max'=~1+kw_min_max+s(kw_min_max),
  'n_tokens_content'=~1+n_tokens_content+s(n_tokens_content),
  'num_hrefs'=~1+num_hrefs+s(num_hrefs),
  'LDA_03'=~1+LDA_03+s(LDA_03),
  'kw_avg_min'=~1+kw_avg_min+s(kw_avg_min),
  'n_non_stop_unique_tokens'=~1+n_non_stop_unique_tokens+s(n_non_stop_unique_tokens),
  'kw_avg_max'=~1+kw_avg_max+s(kw_avg_max),
  'kw_max_max'=~1+kw_max_max+s(kw_max_max),
  'num_self_hrefs'=~1+num_self_hrefs+s(num_self_hrefs),
  'global_rate_positive_words'=~1+global_rate_positive_words+s(global_rate_positive_words),
  'global_subjectivity'=~1+global_subjectivity+s(global_subjectivity),
  'kw_min_min'=~1+kw_min_min+s(kw_min_min),
  'min_positive_polarity'=~1+min_positive_polarity+s(min_positive_polarity),
  'num_keywords'=~1+num_keywords+s(num_keywords),
  'data_channel_is_bus'=~1+data_channel_is_bus,
  'min_negative_polarity'=~1+min_negative_polarity+s(min_negative_polarity),
  'title_sentiment_polarity'=~1+title_sentiment_polarity+s(title_sentiment_polarity),
  'title_subjectivity'=~1+title_subjectivity+s(title_subjectivity),
  'abs_title_subjectivity'=~1+abs_title_subjectivity+s(abs_title_subjectivity)
))

gampred = predict(gamfit, newsTestAll2)
gamacc = ifelse(gampred >= 0.5, 1, 0)
confusionMatrix(factor(gamacc), factor(newsTestAll2$popularity))
plot(gamfit, se = T, ask = T)
