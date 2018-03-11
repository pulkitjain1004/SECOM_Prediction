library(ISLR)
library(MASS)
library(boot)
library(class)
library(psy)
library(randomForest)
library(tree)
library(hmeasure)
library(ROCR)
library(e1071)
library(DiscriMiner)
library(gbm)
library(party)
library(pls)
library(ipred)
library(caret)

setwd("C:/Users/Pulkit Jain/Desktop/ISEN__/613 Engg Data Analysis/Project")
data <- read.csv("project613.csv")
data$y[data$y==-1]<-0
dim(data)
data3 <- data[ lapply(data, function(x) sum(is.na(x))/length(x))<.1]

dim(data3)
data4 <- data[ lapply(data, function(x) sum(is.na(x))/length(x))<.9]
dim(data4)
data5 <- na.omit(data3)
dim(data5)


data7 <- data[ lapply(data, function(x) sum(is.na(x))/length(x))<.5]
dim(na.omit(data4))
# [1] 461 563

data00 <- data[ lapply(data, function(x) sum(is.na(x))/length(x))<0]
dim(na.omit(data00))

# decide to use 10%
# use data5 for PCA

apply(data5, 2, mean)

data6 <- data5[,sapply(data5, function(v) var(v, na.rm=TRUE)!=0)]
# remove columns with zero variance (constant value)
dim(data6)

pr.out <- prcomp(data6[,-1], scale = T)

biplot(pr.out, scale=0)
# too many variables, can't interpret

pr.var=pr.out$sdev^2

pve=pr.var/sum(pr.var)
pve

plot(pve[1:50], xlab="Principal Component", ylab="Proportion of
Variance Explained", ylim=c(0,0.1),type='b')

plot(cumsum(pve), xlab="Principal Component",
     ylab="Cumulative Proportion of Variance Explained",
     ylim=c(0,1),type='b')

# 80% variance will imply nearly 100 variables
# Through scree plot we see using 10 variables is good enough

sum(pve[1:10])
# 28% variance included 

# scree.plot(data6[1:50], type = "V")
# sum(pve[1:5])

dim(pr.out$x)

after.pca <- pr.out$x[,1:15]

dim(after.pca)


after.pca = data.frame(data6[,1], after.pca)
colnames(after.pca)[1] <- "response"

dim(after.pca)

head(after.pca)

# in response 1 corresponds to failure

after.pca$response = as.factor(after.pca$response)

# use after.pca for further analysis

plot(after.pca[,2])
plot(after.pca[,3])
plot(after.pca[,4])
plot(after.pca[,5])
plot(after.pca[,6])
plot(after.pca[,7])
plot(after.pca[,8])

# split in train and test, ratio 75:25




set.seed(1004)
train <- sort(sample(nrow(after.pca), 0.75*nrow(after.pca), replace = F))

data1_train <- after.pca[train,]
data1_test <- after.pca[-train,]

# III.2 tree

tree1 <- tree(response~., data1_train)
plot(tree1)
text(tree1, pretty=0)
summary(tree1)

tree.pred1 <- predict(tree1, data1_test, type="class")  
table(Predicted = tree.pred1, Actual = data1_test$response)
mean(tree.pred1 == data1_test$response)

# tree pruning

set.seed(1004)
cv.tree1 <- cv.tree(tree1, FUN= prune.misclass)
plot(cv.tree1$size, cv.tree1$dev, type= "b")

plot(cv.tree1$size, c(110, 110, 110, 81, 105), type= "b", ylab = "cv.tree1$dev")

cv.tree1$dev
cv.tree1$size

# Check it
min(cv.tree1$dev)
which.min(cv.tree1$dev)
best_tree_size <- cv.tree1$size[which.min(cv.tree1$dev)]
best_tree_size = 4

prune1 <- prune.misclass(tree1, best = best_tree_size)
summary(prune1)



plot(prune1)
text(prune1, pretty= 0)
# the first split is made on alcohol (shows its importance), 
# the other variables are volatile acidity, total sulfur dioxide, sulphates and chlorides


prune.pred1 <- predict(prune1, data1_test, type="class")
table(Predicted = prune.pred1, Actual = data1_test$response)
mean(prune.pred1 == data1_test$response)


# Bagging

set.seed(1004)
bag.wine <- randomForest(response~., data=data1_train, mtry = 10, ntree =200,
                         importance=T)
bag.wine

yhat.bag <- predict(bag.wine, newdata = data1_test )
table(Predicted = yhat.bag,  Actual = data1_test$response)
mean(yhat.bag == data1_test$response)


# Random Forest

err.rf = matrix(0,1,10)

for(i in 1:10){
  
  set.seed(5)
  rf.wine <- randomForest(response~., data=data1_train, mtry = i, ntree =200,
                          importance=T)
  yhat.rf <- predict(rf.wine, newdata = data1_test )
  table(Predicted = yhat.rf, Actual =  data1_test$response)
  err.rf[i] = 1- mean(yhat.rf == data1_test$response)
}
plot(1:10, err.rf, type="b")

err.rf

# since first 5 values are same we choose sqrt(10) = 3

no_pred = 4

rf.wine <- randomForest(response~., data=data1_train, mtry = no_pred, 
                        ntree =100,  importance=T)
importance(rf.wine)
varImpPlot(rf.wine)

yhat.rf <- predict(rf.wine, newdata = data1_test )
table(Predicted = yhat.rf,  Actual = data1_test$response)
mean(yhat.rf == data1_test$response)

# Boosting

set.seed(1004)
boost.wine <- gbm(response~., data = data1_train, distribution = "multinomial"
                  ,n.trees = 5000, interaction.depth = 4)
graphics.off()
# multinomial distribution has been used, which is generalized form of binomial
# due to some error in binomial syntax

summary(boost.wine)

yhat.boost <- predict(boost.wine, newdata = data1_test, n.trees = 5000)

yyhat.boost <- apply(yhat.boost, 1, which.max)

yyhat.boost[yyhat.boost==1] = 0
yyhat.boost[yyhat.boost==2] = 1
table(Predicted = yyhat.boost, Actual =  data1_test$response)
mean(yyhat.boost == data1_test$response)


# KNN


# train independent variables
train_var <- data1_train[,-1]


# test independent variables
test_res <- data1_test[,-1]

# train dependent variables
train_res <- data1_train$response

err <- matrix(0,1,50)
for (i in 1:50){
  set.seed(1004)
  results = knn.cv(train_var, k=i, cl = train_res )
  err[i] = 1-mean(results == train_res)
}

plot(1:50, err, type = "b")
best_knn = which.min(err)
best_knn
# KNN works best when K( no. of nearest neighbours) is 11

knn.pred1 <- knn(train_var, test_res, train_res, k = best_knn)
table(Predicted = knn.pred1, Actual = data1_test$response)
mean(knn.pred1 == data1_test$response)


# # SVM
# Linear

svmfit1 = svm(response~., data1_train, kernel= "linear", cost=.1, scale = F)
summary(svmfit1)

set.seed(1004)
tune.out <- tune(svm, response~., data = data1_train, kernel = "linear",
                 ranges = list(cost=c(.001, .01, .1, 1, 5, 10, 100) ))
summary(tune.out)

tune.out$best.parameters
bestmod = tune.out$best.model


ypred = predict(bestmod, data1_test)

table(Predicted = ypred, data1_test$response)
mean(ypred == data1_test$response)

# 
# svm.radial

svmfit.radial <- svm(response~., data=data1_train, kernal = "radial", 
                     gamma=1, cost = 10 ) 
summary(svmfit.radial)

set.seed(1004)
tune.out.radial <- tune(svm, response~., data= data1_train, kernal = "radial", 
                        ranges = list(cost = 10^(seq(-1,3)), 
                                      gamma = 0.5*(seq(1,5)) ) )

summary(tune.out.radial)

tune.out.radial$best.parameters
tune.out.radial$best.performance

bestmod_rad = tune.out.radial$best.model

ypred_rad = predict(bestmod_rad, data1_test)

table(Predicted = ypred_rad,  Actual = data1_test$response)
mean(ypred_rad == data1_test$response)

# polynomial

svmfit.poly2 <- svm(response~., data=data1_train, kernal = "polynomial", 
                    degree=2, cost = 10 ) 
summary(svmfit.poly2)

set.seed(1004)
tune.out.poly <- tune(svm, response~., data=data1_train, kernal = "polynomial", 
                      ranges = list(cost = 10^(seq(-1,3)), 
                                    degree = c(2,3,4,5,10)) )
summary(tune.out.poly)

tune.out.poly$best.parameters
tune.out.poly$best.performance

bestmod_poly <- tune.out.poly$best.model

ypred_poly = predict(bestmod_poly, data1_test)

table(Predicted = ypred_poly,  Actual = data1_test$response)
mean(ypred_rad == data1_test$response)

# lda
lda.fit <- lda(response~., data=data1_train)
lda.pred <- predict(lda.fit, data1_test)
names(lda.pred)

head(lda.pred)
table(Predicted = lda.pred$class,  Actual = data1_test$response)
mean(lda.pred$class == data1_test$response)

lda.class= rep(0,349)
lda.class[lda.pred$posterior[,2]>0.08] = 1
table(Predicted = lda.class,  Actual = data1_test$response)
# qda

qda.fit <- qda(response~., data=data1_train)
qda.pred <- predict(qda.fit, data1_test)
names(qda.pred)

table(Predicted = qda.pred$class,  Actual = data1_test$response)
mean(qda.pred$class == data1_test$response)

qda.class= rep(0,349)
qda.class[qda.pred$posterior[,2]>0.9] = 1
table(Predicted = qda.class,  Actual = data1_test$response)

# Logistic


logistic.fit=glm(response~.,data=data1_train, family=binomial)
summary(logistic.fit)

logistic.probs=predict(logistic.fit,data1_test,type="response")

logistic.pred=rep("0",nrow(data1_test))
logistic.pred[logistic.probs>0.5]="1"

table(logistic.pred,data1_test$response)
rstandard(logistic.fit)
plot(rstandard(logistic.fit))

# Naive Bayes

naive <- naiveBayes(response~., data = data1_train, type= "response")
npred <- predict(naive, data1_test)

table(Predicted = npred,  Actual = data1_test$response)

