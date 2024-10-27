# Charger les bibliothèques nécessaires
library(psych) # Pour des fonctions psychométriques
library(caTools) # Pour des fonctions de manipulation de données
library(ggplot2) # Pour des visualisations graphiques
library(cowplot) # Pour combiner des graphiques ggplot2
library(ggcorrplot) # Pour visualiser des matrices de corrélation
library(caret) # Pour des fonctions de machine learning
library(pscl) # Pour des modèles régressifs et autres
library(pROC) # Pour analyser les courbes ROC
library(ROSE) # Pour équilibrer les classes dans les données
library(tidymodels) # Pour des workflows de modélisation
library(glmnet) # Pour des modèles de régression pénalisée

# Charger les données
mydata <- read.csv("framingham.csv" , header = TRUE , sep=",") 
head(mydata) # affiche le head 
summary(mydata) # Résumé des données après imputation

str(mydata)
# Vérifier les valeurs manquantes
colSums(is.na(mydata))
# Visualiser les données avec des boxplots
boxplot(mydata$education)
boxplot(mydata$cigsPerDay) 
boxplot(mydata$BPMeds)     
boxplot(mydata$totChol)    
boxplot(mydata$BMI)        
boxplot(mydata$heartRate)  
boxplot(mydata$glucose) 

# Imputer les valeurs manquantes avec des mesures de tendance centrale
mydata$education <- ifelse(is.na(mydata$education),mean(mydata$education,na.rm=TRUE),mydata$education)
mydata$cigsPerDay <-ifelse (is.na(mydata$cigsPerDay),median(mydata$cigsPerDay,na.rm=TRUE),mydata$cigsPerDay)
mydata$BPMeds  <- ifelse (is.na(mydata$BPMeds),median(mydata$BPMeds,na.rm=TRUE),mydata$BPMeds)  
mydata$totChol<- ifelse (is.na(mydata$totChol),median(mydata$totChol,na.rm=TRUE),mydata$totChol)
mydata$BMI     <- ifelse (is.na(mydata$BMI),median(mydata$BMI,na.rm=TRUE),mydata$BMI)
mydata$heartRate <- ifelse (is.na(mydata$heartRate),median(mydata$heartRate,na.rm=TRUE),mydata$heartRate)
mydata$glucose   <- ifelse (is.na(mydata$glucose),median(mydata$glucose,na.rm=TRUE),mydata$glucose)

#Vérifier de nouveau les valeurs manquantes
colSums(is.na(mydata))

# Résumé des données après imputation
summary(mydata)

#Relationship between TenYearCHD and Age / TotCHOL
x <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = age, fill = TenYearCHD)) +geom_boxplot()
y <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = totChol, color = TenYearCHD)) +geom_boxplot()
p <- plot_grid(x, y) 
title <- ggdraw() + draw_label("1. Relationship between TenYearCHD and Age / TotCHOL", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Relationship between TenYearCHD and sysBP / diaBP
x <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = sysBP, fill = TenYearCHD)) +geom_boxplot()
y <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = diaBP, color = TenYearCHD)) +geom_boxplot()
p <- plot_grid(x, y) 
title <- ggdraw() + draw_label("2. Relationship between TenYearCHD and sysBP / diaBP", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Relationship between TenYearCHD and BMI / HeartRate
x <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = BMI, fill = TenYearCHD)) +geom_boxplot()
y <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = heartRate, color = TenYearCHD)) +geom_boxplot()
p <- plot_grid(x, y) 
title <- ggdraw() + draw_label("3. Relationship between TenYearCHD and BMI / HeartRate", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Relationship between TenYearCHD and Glucose / cigsPerDay
x <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = glucose, fill = TenYearCHD)) +geom_boxplot()
y <- ggplot(data = mydata, mapping = aes(x = as.factor(TenYearCHD), y = cigsPerDay, fill = TenYearCHD)) +geom_boxplot()
p <- plot_grid(x,y) 
title <- ggdraw() + draw_label("4. Relationship between TenYearCHD and Glucose / cigsPerDay", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Relationship between TenYearCHD and Sex / Diabetes
x <- ggplot(data = mydata) +geom_count(mapping = aes(x = male, y = TenYearCHD))
y <- ggplot(data = mydata) +geom_count(mapping = aes(x = diabetes, y = TenYearCHD))
p <- plot_grid(x, y) 
title <- ggdraw() + draw_label("5. Relationship between TenYearCHD and Sex / Diabetes", fontface='bold')
plot_grid(title, p, ncol=1, rel_heights=c(0.1, 1))

#Relationship between ageSystolic Blood Pressure
plot(mydata$age, mydata$sysBP, main = "5. Relationship between ageSystolic Blood Pressure",
     xlab = "Age", ylab = "Systolic Blood Pressure")
points(mydata$age, mydata$sysBP, col = ifelse(mydata$male == 0, "blue", "red"), pch = 19)
abline(h = 190, col = "black")
legend("topleft", 
       inset = 0.05,
       legend = c("male", "female"),
       col = c("blue", "red"),
       bg = "gray",
       lwd = c(6, 6))

#Visualiser la matrice de corrélation
corr <- cor(mydata[c(-1,-4)])
ggcorrplot(corr,hc.order = TRUE, type = "lower",lab = TRUE)


#Diviser les données en ensembles d'entraînement et de test
set.seed(123)
split <- sample.split(mydata$TenYearCHD, SplitRatio = 0.8)
train <- subset(mydata, split == TRUE)
test <- subset(mydata, split == FALSE)
dim(train)
dim(test)


# Construire un modèle de régression logistique
model <- glm(formula = TenYearCHD ~.,family='binomial',data = train)
summary(model)

# Prédire sur les données de test
test_pred <- predict(model , newdata = test , type = "response")

#évaluer la performance avec une courbe ROC
roc_obj <- roc(test$TenYearCHD , test_pred)
plot(roc_obj, main = "ROC Curve", col = "blue")
lines(0:1, 0:1, col = "red", lty = 2)
legend("bottomright", legend = paste("AUC =", round(auc(roc_obj), 2)), col = "blue")


# Calculer la matrice de confusion
conf_matrix <- confusionMatrix(as.factor(ifelse(test_pred >= 0.5, 1, 0)), as.factor(test$TenYearCHD))
conf_matrix

# Créer un modèle de régression logistique pénalisée avec tidymodels
train$TenYearCHD<- as.factor(train$TenYearCHD)
test$TenYearCHD<- as.factor(test$TenYearCHD)
model2 <- logistic_reg(mixture = double(1), penalty = double(1)) %>%
  set_engine("glmnet") %>%
  set_mode("classification") %>%
  fit(TenYearCHD ~ ., data = train)

tidy(model2)

# Prédire sur les données de test
predictions <- predict(model2, new_data = test, type = "prob") %>%
  mutate(prediction = ifelse(.pred_1 > 0.5, 1, 0))  # Assuming threshold of 0.5 for binary classification

predictions$prediction <- factor(predictions$prediction, levels = c(0, 1))
test$TenYearCHD <- factor(test$TenYearCHD, levels = c(0, 1))


#ROC Curve
roc_curve <- roc(as.numeric(predictions$prediction), as.numeric(test$TenYearCHD))
plot(roc_curve, main = "ROC Curve", col = "blue")
lines(0:1, 0:1, col = "red", lty = 2)
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 2)), col = "blue")

# Calculer la matrice de confusion
conf_matrix <- confusionMatrix(predictions$prediction, test$TenYearCHD)
conf_matrix




