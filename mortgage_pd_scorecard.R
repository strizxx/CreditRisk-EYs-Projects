#install.packages("scorecard")
#install.packages("dplyr")
#install.packages("corrplot")
#install.packages("ROCit")
#install.packages("VIM")
#install.packages("MASS")

library(scorecard)
library(dplyr)
library(corrplot)
library(ROCit)
library(VIM)
library(MASS)

path = "your path"
setwd(path)
data <- read.csv("mortgage_sample.csv")




##################### SEPERATING PUBLIC SAMPLES #####################

public_data <- data[data$sample=="public",]
dim(public_data)


# We came across several duplicated rows, e.g. id=37130 & time=51
# So, we're selecting only distinct observations
public_data <- distinct(public_data)
dim(public_data)

# 276 duplicate rows were deleted

# resetting index
rownames(public_data) <- 1:nrow(public_data) 

#____________________________________________________________________







##################### MISSING VALUES TREATMENT #####################
sum(is.na((public_data)))
# 173 observations have NA in at least 1 column

summary(public_data)
# it can be observed that all NA values lie in LTV_time column

# NA's are replaced by the result from the median calculation of 8th nearest neighborhood
# in the data set we have 560888 observations, and 173 missing value, in this case we took average frequency of missing data(560888/173)=3242,
# then use sqrt value as K to prevent over fitting by small k and under fitting by large k,as sqrt(560888/173=57),57 is the final value we used in KNN imputation  
public_data <- kNN(public_data, variable="LTV_time" , k=57)[,-25]

#____________________________________________________________________








##################### CREATING TARGET AND COHORTS #####################

# recoding 2 in status time as 0 (pay-off means non-default for our purposes)
public_data$status_time[public_data$status_time == 2] <- 0

# 1. CREATE TARGET COLUMN
#    We assign target value only to every 12th row for each id, 
#    all other rows are set to NA and then filtered out
#    Also, if id has only 1 observation (it has no history in the data), 
#    it will be filtered out as well (target will be NA) - e.g. id 8
#    The cases where an id has only 13/25/... observations are treated similarly: 
#    the last observation will be assigned NA, as there is no data for the next 12 months - e.g. id 9
target <- c()

for (id in unique(public_data$id)) { # The loop takes approx.3-4 minutes
  temp <- public_data$status_time[public_data$id==id]
  rows <- length(temp)
  target.id <- rep(NA, rows)
  
  i <- 1
  while (i < rows) {
    target.id[i] <- sum(temp[(i+1):min((i+12), rows)])
    i <- i + 12
  }
  
  target <- append(target, target.id)
}

public_data$target <- target

#View(public_data[, c(1,2,23,25)])

# 2. CREATE SAMPLE FOR MODELLING - TAKE EVERY 12TH OBSERVATION (all other observations have target==NA)
sample <- public_data[!is.na(public_data$target), ]

# Now all rows in our sample must have target either 0 or 1. Check it:
# View(data[data$id %in% sample$id[sample$target>1], ])
# Data for id 3014 are corrupted (it contains 2 different rows for the same time value).
# We exclude it:
sample <- sample[sample$id!=3014, ]

#_________________________________________________________________________________________







##################### OUTLIERS TREATMENT #####################


# set the boundaries on 5th and 95th quantile, set outliers = values on boundaries
fun <- function(x){
  quantiles <- quantile( x, c(.05, .95 ) )
  x[ x < quantiles[1] ] <- quantiles[1]
  x[ x > quantiles[2] ] <- quantiles[2]
  return(x)
}


for(i in 1:ncol(sample[,-c(1:5,12:15,21:25)])) {
  sample[,-c(1:5,12:15,21:25)][,i] <- fun(sample[,-c(1:5,12:15,21:25)][,i])
}

summary(sample)

#_________________________________________________________________________________________







##################### FEATURE ENGINEERING & TRAIN AND TEST SETS #####################
# Creating 2 new variables: time_to_mat & duration
sample$time_to_mat <- sample$mat_time - sample$time
sample$duration <- sample$time - sample$orig_time

# remove redundant columns : 
# "orig_time", "first_time", "maturity_time", 
# "default_time", "payoff_time", "status_time"
# "sample"
names(sample)
sample <- sample[, -c(3:5,21:24)]
names(sample)


# split data into training and testing subsets
set.seed(1000)
prop <- 0.7
train.cases <- sample(nrow(sample),
                      nrow(sample) * prop)
train <- sample[train.cases, ]
test <- sample[-train.cases, ]


# BINNING & WoE
bins.train <- woebin(train[, -c(1:2)], y ="target")
train.woe <- woebin_ply(train, bins.train, to="woe")
test.woe <- woebin_ply(test, bins.train, to="woe")

# Bins Visualization:
woebin_plot(bins.train)

#_________________________________________________________________________________________








##################### UNIVARIATE ANALYSIS #####################
options(scipen = 100) #disable scientific notation for readability

# Calculate IV (information value) for each variable:
iv <- iv(train.woe[,-c(1:2)], y="target")
iv <- iv %>% mutate(pred_power =
                   case_when(info_value < 0.02 ~ "useless", 
                          info_value < 0.1 ~ "weak",
                          info_value < 0.3 ~ "medium",
                          info_value >= 0.3 ~ "strong")
)
iv

#_________________________________________________________________________________________






##################### CORRELATION ANALYSIS #####################
corr <- round(cor(train.woe[, -c(1:3)], use = "pairwise.complete.obs", method="spearman"), 3)

par(cex = 0.7)
corrplot.mixed(corr, upper = "ellipse", tl.pos = "lt", number.cex = 0.7,
               tl.cex = 1/par("cex"),
               cl.cex = 1/par("cex"))

# Groups of variables with correlation > 0.7
cmat <- abs(cor(train.woe[, -c(1:3)], use = "pairwise.complete.obs", method="spearman"))
groups <- lapply(rownames(cmat), function(rname) { colnames(cmat)[cmat[rname, ]>0.7] })
groups <- unique(groups)
groups

#_________________________________________________________________________________________







##################### INITIAL MODEL INPUT #####################
# We remove balance_orig_time, Interest_Rate_orig_time, hpi_time, duration based on corr analysis and IV
names(train.woe)
initial_df_train <- train.woe[,-c(14,17,7,20)]
initial_df_test <- test.woe[,-c(14,17,7,20)]
names(initial_df_train)

#_________________________________________________________________________________________






##################### MODELLING ##################### 

glm.fit <- glm(as.factor(target) ~ .,
               data=initial_df_train[, -c(1:2)], family = 'binomial')

summary(glm.fit)
# the summary shows that the three real estate columns have the highest p values, we move forward with
# step wise regression to dig deeper

# Step wise regression
stepwise <- stepAIC(glm.fit, direction="backward")
stepwise

summary(stepwise)

# AIC quantifies the amount of information loss due to this simplification.
# Hence, the model with the lower AIC value is preferred.
# The model started with three RE columns separated out, contributing the least to the information loss
# However, the last model keeps the Single Family RE column in the list of optimal variables
# Hence, we go ahead and remove only REtype_PU_orig_time_woe and REtype_CO_orig_time_woe columns
#_________________________________________________________________________________________





##################### MODEL EVALUATION #####################
glm.prob <- predict(stepwise, initial_df_test, type="response")
prob.test <- data.frame(glm.prob=glm.prob, target=initial_df_test$target)

aggregate(as.numeric(prob.test$glm.prob) ~ prob.test$target, prob.test, FUN=mean)
# higher probabilities resonate with default


# GINI & CAP Curve
perf = perf_eva(pred=prob.test$glm.prob, label=prob.test$target, binomial_metric="gini", show_plot=c("lz"))
perf$binomial_metric
# the model has the discriminatory power/ GINI coefficient: 0.5157

#_________________________________________________________________________________________








##################### ACCURACY AND ROC ANALYSIS #####################

glm.class <-  glm.prob
glm.class[glm.class > 0.5] <- "Yes"
glm.class[glm.class <= 0.5] <- "No"
#starting with the threshold of 0.5

#Confusion Matrix
(confmatrix <- table(prediction = glm.class, Reality = initial_df_test$target))

#Accuracy Before ROC
(confmatrix[1] + confmatrix[4]) /  sum(confmatrix)
#80%


#ROC analysis
glm.roc <- plot(rocit(glm.prob, initial_df_test$target))
glm.roc

#adjusting the cutoff as suggested by ROC

glm.class2 <-  glm.prob
glm.class2[glm.class2 > 0.1857409   ] <- "Yes"
glm.class2[glm.class2 <= 0.1857409  ] <- "No"



(confmatrix2 <- table(prediction = glm.class2, Reality = initial_df_test$target))


#Accuracy After ROC
(Accuracy_LR <- (confmatrix2[1] + confmatrix2[4]) /  sum(confmatrix2))
#Final accuracy of 67%

#_________________________________________________________________________________________





##################### SCORECARD #####################
card <- scorecard(bins.train, stepwise, points0=600, odds0=1/19, pdo=50,
                  basepoints_eq0=FALSE, digits=0)
card
#_________________________________________________________________________________________






##################### FINAL DATAFRAME: ID, TIME, PD & SCORE #####################
sample.woe <- woebin_ply(sample, bins.train, to="woe")
pd <- predict(stepwise, sample.woe, type="response")
score <- scorecard_ply(sample, card)

final.df <- data.frame(id=sample$id, time=sample$time, target=sample$target, pd=pd, score=score)
write.csv(final.df, paste0(path, "/mortgage_pd.csv"), row.names=FALSE)
View(final.df)


#_________________________________________________________________________________________




