file <- "https://raw.githubusercontent.com/danielcaba889/Tesis-FCE-UBA---Analisis-de-Regresion-Logistica-para-medir-el-riesgo-de-contraer-cancer-de-mama/main/wdbc.data"
breastdf <- read.csv(file, sep=",", header = FALSE)[-1]
View(breastdf)


breastdf_normalize <- data.frame(cbind(classification= breastdf$V2== 'M', scale(breastdf[-1] )))
breastdf_normalize
View(breastdf_normalize)

library(ggplot2)
ggplot(data = breastdf_normalize, aes(x=V3, y =classification))+
  geom_point()
ggplot(data = breastdf_normalize, aes(x=V4, y =classification))+
  geom_point()
ggplot(data = breastdf_normalize, aes(x=V5, y =classification))+
  geom_point()
ggplot(data = breastdf_normalize, aes(x=V6, y =classification))+
  geom_point()
ggplot(data = breastdf_normalize, aes(x=V7, y =classification))+
  geom_point()
ggplot(data = breastdf_normalize, aes(x=V8, y =classification))+
  geom_point()



summary(breastdf)

log_reg <- glm(classification~., data = breastdf_normalize, family = 'binomial',control = list(maxit=100000))

summary(log_reg)
coef(log_reg)
exp(coef(log_reg))
confint(log_reg)


plot(log_reg$residuals)
plot(log_reg)

ggplot(data = breastdf_normalize, aes(x=V3, y = classification))+
  geom_point()

V3_model <- glm(classification~V3, data = breastdf_normalize, family = 'binomial',control = list(maxit=10000))
summary(V3_model)
coef(V3_model)
exp(coef(V3_model))
V4_model <- glm(classification~V4, data = breastdf_normalize, family = 'binomial',control = list(maxit=10000))
summary(V4_model)
coef(V4_model)
exp(coef(V4_model))

full_breast_model <- glm(classification~., data = breastdf_normalize, family = 'binomial',control = list(maxit=10000))
summary(full_breast_model)
coef(full_breast_model)
exp(coef(full_breast_model))
confint(full_breast_model)


library(aod)
wald.test(b = coef(full_breast_model), Sigma = vcov(full_breast_model), Terms = 1:31)

library(caret)

Breast_CrossValSettings <- trainControl(method = 'repeatedcv', number=10, savePredictions = TRUE)

Breast_CrossVal <- train(as.factor(classification)~., data = breastdf_normalize, family='binomial',method='glm',
                  trControl=Breast_CrossValSettings)

breast_pred <- predict(Breast_CrossVal, newdata = breastdf_normalize)
Breast_Conf_Matrix <- confusionMatrix(breast_pred, as.factor(breastdf_normalize$classification))

coeftest(full_breast_model)


new_log_reg <- glm(classification~V26+V13+V29, data = breastdf_normalize, family = 'binomial',control = list(maxit=100000))
summary(new_log_reg)
coef(new_log_reg)
exp(coef(new_log_reg))

