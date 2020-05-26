#clear screen
cat("\014")

dataset <- read.csv("/Users/anuradha/Desktop/Spring/DS for BI/OnlineNewsPopularity/OnlineNewsPopularity.csv")

summary(dataset)

#check total rows
nrow(dataset)

#check for null values
nrow(na.omit(dataset))

str(dataset)

#remove url and timedelta
drop_col =c("url","timedelta")
dataset <- dataset[,-match(drop_col,names(dataset))]

#content with 0 length are invalid (1181 rows)
length(which(dataset$n_tokens_content==0))
dataset <- dataset[!(dataset$n_tokens_content==0),]
nrow(dataset)

#null values
sum(is.na(dataset))

#distribution is right-skewed
hist(news_data$shares,xlab = "#of shares", main="#shares distribution",border="blue", col="yellow")

# 0,1 values
unique(dataset[,c("data_channel_is_world")])
unique(dataset[,c("weekday_is_sunday")])

# convert to categorical for news topic and news day
dataset$data_channel_is_lifestyle <- as.factor(dataset$data_channel_is_lifestyle)
dataset$data_channel_is_entertainment <- as.factor(dataset$data_channel_is_entertainment)
dataset$data_channel_is_bus <- as.factor(dataset$data_channel_is_bus)
dataset$data_channel_is_socmed <- as.factor(dataset$data_channel_is_socmed)
dataset$data_channel_is_tech <- as.factor(dataset$data_channel_is_tech)
dataset$data_channel_is_world <- as.factor(dataset$data_channel_is_world)

dataset$weekday_is_monday <- as.factor(dataset$weekday_is_monday)
dataset$weekday_is_tuesday <- as.factor(dataset$weekday_is_tuesday)
dataset$weekday_is_wednesday <- as.factor(dataset$weekday_is_wednesday)
dataset$weekday_is_thursday <- as.factor(dataset$weekday_is_thursday)
dataset$weekday_is_friday <- as.factor(dataset$weekday_is_friday)
dataset$weekday_is_saturday <- as.factor(dataset$weekday_is_saturday)
dataset$weekday_is_sunday <- as.factor(dataset$weekday_is_sunday)

str(dataset)
summary(dataset)

# check news topic vs shares
boxplot()


# check news day vs shares

#check outliers and remove

#modelling
install.packages('caTools')
library(caTools)
set.seed(1000)
sample_size <- floor(0.7 * nrow(dataset))
train <- sample(seq_len(nrow(dataset)), size = sample_size)
train_set <- dataset[train, ]
test_set <- dataset[-train, ]

#linear model- all variables
lm_fit = lm(shares~., data=train_set)
summary(lm_fit)
lm_predict = predict(lm_fit, data=test_set)
plot(lm_fit)

#RMS error for linear model=17397.4
sqrt(mean((test_set$shares - lm_predict)^2))

#fit linear model with log- all variables
dataset$shares <- log(dataset$shares)
set.seed(1000)
sample_size <- floor(0.7 * nrow(dataset))
train <- sample(seq_len(nrow(dataset)), size = sample_size)
train_set <- dataset[train, ]
test_set <- dataset[-train, ]
log_fit = lm(shares~., data=train_set)
summary(log_fit)
log_predict = predict(log_fit, data=test_set)
#RMS error with log =0.992105
sqrt(mean((test_set$shares - log_predict)^2))
plot(log_fit)

#fit log model with significant variables
sg_fit = lm(shares~n_tokens_title+n_tokens_content+n_non_stop_unique_tokens+num_hrefs+
              num_self_hrefs+average_token_length+num_keywords+data_channel_is_lifestyle+
              data_channel_is_entertainment+data_channel_is_bus+data_channel_is_socmed+data_channel_is_tech+
              data_channel_is_world+kw_min_min+kw_max_min+kw_avg_min+kw_min_max+kw_avg_max+kw_min_avg+kw_max_avg+kw_avg_avg
            +self_reference_avg_sharess+weekday_is_monday+weekday_is_tuesday+weekday_is_wednesday+weekday_is_thursday+weekday_is_friday
            +LDA_00+LDA_01+LDA_02+LDA_03+global_subjectivity+global_rate_positive_words+min_positive_polarity+
              title_subjectivity+title_sentiment_polarity+abs_title_subjectivity, data=train_set)
#backward step
step(sg_fit)
summary(sg_fit)
sg_predict = predict(sg_fit, data=test_set)
#RMS error with sg =0.9916583
sqrt(mean((test_set$shares - sg_predict)^2))


#check multicolinearity, AIC, vif, anova test



