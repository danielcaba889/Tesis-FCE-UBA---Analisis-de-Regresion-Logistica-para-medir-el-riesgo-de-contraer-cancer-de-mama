# tesis
codigo en R y Python

##### Importo librerias ###########
library(caret)
library(dplyr)
library(MASS)
library(GGally)
library(olsrr)
library(MLmetrics)

########## link de acceso a base de datos ###########
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

###### importo datos y los limpio ######
excel <- choose.files()
df <- read.csv(excel,header=F)
View(df)
colnames(df) <- c("ID","Diagnosis","radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean", "compactness_mean","concavity_mean","concave_points_mean", "symmetry_mean",   "fractal_dimension_mean","radius_se","texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se","radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst")
df["Diagnosis"] <- df["Diagnosis"] == "M"
########## Divido en set de entrenamiento y testeo ###########
set.seed(1234) #randomization`
#creating indices
trainIndex <- createDataPartition(df$Diagnosis,p=0.7,list=FALSE)

#splitting data into training/testing data using the trainIndex object
train <- df[trainIndex,] #training data (70% of data)
test <- df[-trainIndex,] #testing data (30% of data)

colnames(train)
########## LOGISTIC REGRESSION ########
LogReg1 <- glm(Diagnosis~get(colnames(train)[2]), family = 'binomial',
               data = train, maxit=100000)

for(i in 2:ncol(train)){
  vble <- colnames(train)[i]
  logreg1 <- glm(Diagnosis~get(vble),family = 'binomial',
                 data = train, maxit=10000000)
  print(colnames(train)[i])
  print( summary(logreg1))
}
# muchas variables con pvalue <2e-16 ***
#las elijo para el subset

#radius_mean texture_mean perimeter_mean area_mean
#smoothness_mean compactness_mean concavity_mean concave_points_mean
#radius_se perimeter_se area_se radius_worst 
# texture_worst perimeter_worst area_worst 
#compactness_worst concavity_worst concave_points_worst

LogReg2 <- glm(Diagnosis~radius_mean+ texture_mean+ perimeter_mean+ area_mean+
               smoothness_mean+ compactness_mean+ concavity_mean+ concave_points_mean+
               radius_se+ perimeter_se+ area_se+ radius_worst+ 
                texture_worst+ perimeter_worst+ area_worst+ 
               compactness_worst+ concavity_worst+ concave_points_worst, family = 'binomial',
               data = train, maxit=100000)
summary(LogReg2) #casi ninguna significativa
#saco las menos significativas

LogReg3 <- glm(Diagnosis~radius_mean+ texture_mean+ perimeter_mean+ area_mean+
                 smoothness_mean+ compactness_mean+ concavity_mean+ concave_points_mean+
                 radius_se+ perimeter_se+ area_se+ 
                 texture_worst+ perimeter_worst+  
                  concavity_worst+ concave_points_worst, family = 'binomial',
               data = train, maxit=100000000)
summary(LogReg3)
#algunos empiezan a ser significativos
#los reuno
LogReg4 <- glm(Diagnosis~smoothness_mean+compactness_mean+
                 concavity_mean+concave_points_mean+
                 perimeter_se+perimeter_worst+concavity_worst,
               family = 'binomial',
               data = train, maxit=100000000)
summary(LogReg4)
#saco el menos significativo
LogReg5 <- glm(Diagnosis~smoothness_mean+compactness_mean+
                 concave_points_mean+
                 perimeter_se+perimeter_worst+concavity_worst,
               family = 'binomial',
               data = train, maxit=100000000)
summary(LogReg5)
#saco el menos significativo de nuevo
LogReg5 <- glm(Diagnosis~smoothness_mean+compactness_mean+
                 perimeter_worst+concavity_worst,
               family = 'binomial',
               data = train, maxit=100000000)
summary(LogReg5)
########## CROSS VALIDATION #############
# define training control
set.seed(115)
train_control <- trainControl(method = "cv", number = 5)

# train the model on training set
cv_model <- train(as.character(Diagnosis)~smoothness_mean+compactness_mean+
                 perimeter_worst+concavity_worst,
               data = train,
               trControl = train_control,
               method = "glm",
               family="binomial",maxit=100000000000)
cv_model
cv_model$resample
# mido el f1
set.seed(115)
train_control <- trainControl(method = "cv", number = 5)
# train the model on training set
cv_model <- train(as.character(Diagnosis)~smoothness_mean+compactness_mean+
                    perimeter_worst+concavity_worst,
                  data = train,
                  trControl = train_control,
                  method = "glm",
                  family="binomial",metric="F1")
cv_model
cv_model$resample
# print cv scores
summary(cv_model)

############## PREDICCIONES ###########
mean(ifelse(LogReg5%>% predict(test, type="response")>0.5, T,F)==test$Diagnosis)
#Aumentó la precisión en set de testeo

step_pred <- predict(LogReg5,test,type="response")>0.5 #predicciones
colnames(step_pred) <- "Diagnosis"
mean(step_pred==test$Diagnosis) # 95.29% accuracy en set de testeo
######### CONFUSION MATRIX ##########
confusionMatrix(factor(step_pred), as.factor(test$Diagnosis),positive = "TRUE")

#Calculo el F1-SCORE
step_f1 <- F1_Score(test$Diagnosis, step_pred, positive = "TRUE")

########### ESPERANZA DE LA CANTIDAD DE SINIESTROS #########
stros_esperados <- sum(predict(LogReg5,test, type="response"))
sum(test$Diagnosis) #stros reales
npacientes <- nrow(test) #cant pacientes

stros_esperados/npacientes #P(M/C)

########### ROC Y AUC ###############
library(pROC)
library(ROCR)
pr_model1 <- predict(LogReg5)
pr_model2 <- prediction(pr_model1,test$Diagnosis)
plot(performance(pr_model2, "tpr","fpr"), lwd=2, col="purple")
abline(a=0,b=1, lty="dashed")

auc_ROCR <- performance(pr_model2, measure = "auc")
auc_ROCR <- auc_ROCR@y.values[[1]]
5.2.2	Árbol de decision
tree = DecisionTreeClassifier(random_state=0,max_depth=2)
tree.fit(X_train, y_train)
tree_predict = tree.predict(X_test)
from sklearn.tree import plot_tree
fig = plt.figure(figsize=(10,10))
plot_tree(tree, filled=True, rounded=True,fontsize=15,class_names=["True","False"] ,feature_names = X_train.columns)
print("f1 score en set de testeo: " + str(f1_score(y_test, tree_predict)))
tree.score(X_train, y_train)
kfolds = ["Fold 1","Fold 2","Fold 3", "Fold 4","Fold 5"]
resultados_cv = pd.DataFrame({"Fold":kfolds,
                             "score":tree_cross_val,
                             "f1 score":tree_f1_cross_val})

tabla_score_promedio = pd.DataFrame({"Medida":["Score promedio","f1 score promedio", "Score set de prueba", "f1 score set de prueba"],
  "Valor":[tree_cross_val.mean(), tree_f1_cross_val.mean(),tree.score(X_test, y_test),
           f1_score(y_test, tree_predict)]})

print(resultados_cv,"\n")
print(tabla_score_promedio)
plot_confusion_matrix(tree, X_train, y_train)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
tree_cm = confusion_matrix(y_true=y_test, y_pred=tree_predict, labels=tree.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm,
                               display_labels=tree.classes_)
disp.plot() 
#tree_predict
from sklearn.metrics import accuracy_score
tree_test_score = accuracy_score(y_true=y_test, y_pred=tree_predict)
tree_recall = recall_score(y_test, tree_predict)
tree_precision = precision_score(y_test, tree_predict)
tree_f1_test = f1_score(y_test, tree_predict)


tree_recall_precision_table = {"Medida":["Score","F1-score","Recall","Precision",],
                               "Valores":[tree_test_score,tree_f1_test,tree_recall,tree_precision]}
pd.DataFrame(tree_recall_precision_table)
5.2.3	Análisis de componentes principals
breast_pca = PCA(n_components = 2)
X_2D = breast_pca.fit_transform(X)
breast_pca.explained_variance_ratio_
tabla_varianza_explicada = pd.DataFrame({
    "PC":["PC1","PC2"],
    "Explained variance ratio":[breast_pca.explained_variance_ratio_[0], 
                                breast_pca.explained_variance_ratio_[1]]
})
tabla_varianza_explicada
pc1 = X_2D[:,0]
pc2 = X_2D[:,1]
plt.figure(figsize = (8,4))
plt.scatter (pc1,pc2, c=y, cmap='coolwarm',alpha=0.75)
plt.xlabel("PC1")
plt.ylabel("PC2")
pca = PCA()
pca.fit(X)

number_of_dimensions = np.arange(1,31)
variance_cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(number_of_dimensions,variance_cumsum, c="purple", linewidth=2)
plt.scatter([np.argmax(variance_cumsum>=0.983)+1], [0.99822], s=20, c="blue")
