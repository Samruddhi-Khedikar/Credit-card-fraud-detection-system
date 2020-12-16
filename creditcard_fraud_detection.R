#importing the dataset
creditcard_data <- read.csv("creditcard.csv")

#structure of the dataset
str(creditcard_data)

#converting the class to factor class
creditcard_data$Class<- factor(creditcard_data$Class,levels=c(0,1))

dim(creditcard_data)
head(creditcard_data,6)
tail(creditcard_data,6)

#to get the distribution of fraud and legit transactions
table(creditcard_data$Class)

#summary
summary(creditcard_data)

#to count the number of null values 
sum(is.na(creditcard_data))

#to calculate the percentage of legit and fraud transactions
prop.table(table(creditcard_data$Class))


#pie chart of credit card transactions
lables <- c("legit","fraud")
lable <- paste(lables , round(100*(prop.table(table(creditcard_data$Class))) , 2))
final <-paste0(lable,"%")
final
pie(table(creditcard_data$Class),final,col = c("orange","red"), main = "Pie chart of credit card transactions")

#no model prediction
predictions <- rep.int(0,nrow(creditcard_data))
predictions <- factor(predictions, level= c(0,1))
library(caret)
confusionMatrix(data= predictions, reference= creditcard_data$Class)


#we create a model to fasten the computation
library(dplyr)

set.seed(1)
creditcard_data <- creditcard_data %>%sample_frac(0.1)
table(creditcard_data$Class)

library(ggplot2)
ggplot(data= creditcard_data, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  scale_color_manual(values=c('dodgerblue2','red'))

  
#----------------------------------creating train and test sets for fraud detection model--------------------------------------------------------------#
library(caTools)
set.seed(123)
data_sample= sample.split(creditcard_data$Class, SplitRatio = 0.8)
train_data=subset(creditcard_data,data_sample==TRUE)
test_data= subset(creditcard_data,data_sample==FALSE)
dim(train_data)
dim(test_data)


#using SMOTE to balance the data
library(smotefamily)
table(train_data$Class)

#set the number of fraud and legitimate cases, and the desired percentage of legitimate cases
n0<-22750
n1<- 35
r0<-0.6 

#calculate the value for dup_size parameter of SMOTE
ntimes <-((1-r0)/r0)*(n0/n1)-1

smote_output= SMOTE(X=train_data[,-c(1,31)],target = train_data$Class,K=5,dup_size = ntimes)
credit_smote <- smote_output$data
colnames(credit_smote)[30]<-"Class"
prop.table(table(credit_smote$Class))

#--------------------------------------------class distribution for original dataset-----------------------------------------------------------------
ggplot(train_data,aes(x=V1,y=V2,color= Class))+
  geom_point()+
  #theme_bw()
  
  scale_color_manual(values=c('dodgerblue2','red'))

#-------------------------------------------class distribution for sample dataset using smote---------------------------------------------
ggplot(credit_smote,aes(x=V1,y=V2,color= Class))+
  geom_point()+
  scale_color_manual(values=c('dodgerblue2','red'))


#decision tree to predict if data is fraud or legitimate
library(rpart)
library(rpart.plot)

CART_model<-rpart(Class~ ., credit_smote)
rpart.plot(CART_model, extra=0,type=5, tweak=1.2)


#predict fraud classes
predicted_val <- predict(CART_model, test_data,type='class')
#build confusion matrix
confusionMatrix(predicted_val, test_data$Class)

#--------------------------------------with smote data prediction for whole dataset--------------------------------------------------------
predicted_val<- predict(CART_model,creditcard_data[,-1],type = 'class')
confusionMatrix(predicted_val,creditcard_data$Class)

#decision tree without the smote data

CART_model<-rpart(Class~ ., train_data[,-1])

rpart.plot(CART_model, extra=0,type=5, tweak=1.2)
predicted_val <- predict(CART_model, test_data[-1],type='class')

confusionMatrix(predicted_val, test_data$Class)

#--------------------------------------without smote whole data prediction-------------------------------------------------------------------
predicted_val <- predict(CART_model, creditcard_data[,-1],type='class')
confusionMatrix(predicted_val,creditcard_data$Class)

