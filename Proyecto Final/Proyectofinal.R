############################################
#######PROYECTO FINAL################


##Parkinsons Monitoring######

##Desarrollado por: Adrián Sánchez Cerrillo y Miguel Ámgel López Robles###

##############################################

library("caret")
library("leaps")
library("glmnet")
library("MASS")
library("randomForest")
library("nnet")
library("mboost")
library("kernlab")


#establecer la semilla
set.seed(3)
#lectura de datos
datos = read.csv("./datos/parkinsons.csv")

# comprobar si hay datos perdidos
perdidos = sum(is.na(datos))

print("vemos el número de datos perdidos")
print(perdidos)
scan(n=1)


#eliminar las 4 primeras columnas que no se usan para la regresión
datos = datos[,-c(1,2,3,4)]

#extraer las dos variables que debemos predecir
UPDRSmotor = datos[,1]
UPDRStotal = datos[,2]
datos = datos[,-c(1,2)]


#definir un conjunto de test
indices.train = sample(nrow(datos),size = nrow(datos)*0.8)
datos.train = datos[indices.train,]
datos.test = datos[-indices.train,]

UPDRSmotor.train = UPDRSmotor[indices.train]
UPDRSmotor.test = UPDRSmotor[-indices.train]

UPDRStotal.train = UPDRStotal[indices.train]
UPDRStotal.test = UPDRStotal[-indices.train]

###preprocesado de los datos 

prePr = preProcess(datos.train, method = c("center","scale"))
datos.train = predict(prePr,datos.train)
datos.test = predict(prePr,datos.test)

###Probar que nos daria pca

prePrPCA = preProcess(datos.train, method = c("center","scale","pca"))
datos.trainPCA = predict(prePrPCA,datos.train)

print("Si usamos PCA reduciriamos el numero de características a: ")
print(dim(datos.trainPCA)[2])
print("Continuamos sin usar PCA")
rm(datos.trainPCA)
scan(n=1)

##Probar el regsubset

#calcular los mejores subset
subsetMotor = regsubsets(x = datos.train, y = UPDRSmotor.train,
                         nvmax = dim(datos.train)[2], method = "forward")

subsetTotal = regsubsets(x = datos.train, y = UPDRStotal.train,
                         nvmax = dim(datos.train)[2], method = "forward")

summaryMotor = summary(subsetMotor)
summaryTotal = summary(subsetTotal)


#pintar las graficas para ver los resultados para motor 
par(mfrow=c(2,2))

plot(summaryMotor$rss, xlab = "Número de variables.", ylab = "RSS", type = "l",col = "green")
minimo = which.min(summaryMotor$rss)
points(minimo,summaryMotor$rss[minimo], col = "red", pch = 10)

plot(summaryMotor$adjr2, xlab = "Número de variables.", ylab = "RSQajustado", type = "l", col="blue")
maximo = which.max(summaryMotor$adjr2)
points(maximo,summaryMotor$adjr2[maximo], col = "red", pch = 10)

plot(summaryMotor$cp, xlab = "Número de variables.", ylab = "CP", type = "l", col ="grey")
minimo = which.min(summaryMotor$cp)
points(minimo,summaryMotor$cp[minimo], col = "red", pch = 10)

plot(summaryMotor$bic, xlab = "Número de variables.", ylab = "BIC", type = "l", col= "purple")
minimo = which.min(summaryMotor$bic)
points(minimo,summaryMotor$bic[minimo], col = "red", pch = 10)
title("UPDRSmotor",outer = T, line = -2)
print("Plot de los resultados de subset selection para predecir UPDRSmotor")
scan(n=1)

#pintar las graficas para ver los resultados para total 
par(mfrow=c(2,2))

plot(summaryTotal$rss, xlab = "Número de variables.", ylab = "RSS", type = "l",col = "green")
minimo = which.min(summaryTotal$rss)
points(minimo,summaryTotal$rss[minimo], col = "red", pch = 10)

plot(summaryTotal$adjr2, xlab = "Número de variables.", ylab = "RSQajustado", type = "l", col="blue")
maximo = which.max(summaryTotal$adjr2)
points(maximo,summaryTotal$adjr2[maximo], col = "red", pch = 10)

plot(summaryTotal$cp, xlab = "Número de variables.", ylab = "CP", type = "l", col ="grey")
minimo = which.min(summaryTotal$cp)
points(minimo,summaryTotal$cp[minimo], col = "red", pch = 10)

plot(summaryTotal$bic, xlab = "Número de variables.", ylab = "BIC", type = "l", col= "purple")
minimo = which.min(summaryTotal$bic)
points(minimo,summaryTotal$bic[minimo], col = "red", pch = 10)
title("UPDRStotal",outer = T, line = -2)

print("Plot de los resultados de subset selection para predecir UPDRStotal")
print("La decision va a ser no reducir las características")
scan(n=1)

par(mfrow=c(1,1))


####probar con lineales
#control para la funcion train de la libreria caret
control = trainControl(method = "cv", number = 5,allowParallel=TRUE)
LinealSimple<- train(x = datos.train, y = UPDRSmotor.train, method = "lm", trControl = control, metric = "Rsquared")
print("con un modelo lineal simple obtenemos los siguientes resultados" )
print(LinealSimple$results)
print("como podemos ver el Rsquared es bajo y no tenemos un modelo muy bueno vamos a probar con más complejidad")
scan(n=1)



##aumentamos a funciones
if(F){
train = datos.train
maxIt = 7
for (i in 2:maxIt){
  #aumentar la clase
  train = cbind(train,I(datos.train)^i)
  #entrenamos con lasso
  fitLasso <- train(x = train, y = UPDRSmotor.train, 
                    method = "lasso",
                    trControl = control,
                    metric = "Rsquared"
  )
  fitRidge<- train(x = train, y = UPDRSmotor.train, 
                    method = "ridge",
                    trControl = control,
                    metric = "Rsquared"
  )
  
  cat("Polinomio Grado ", i, "para Lasso", "\n")
  print(fitLasso$results)
  
  
  cat("Polinomio Grado ", i,"para Ridge", "\n")
  print(fitRidge$results)
  
}

print("Como podemos ver nuestros modelos lineales han sido insuficientes")
scan(n=1)
}

################################################################
#modelos no lineales
######################################
#vamos a probar boosting random forest y svm

#####random forest
if(T){
      #esto son los resultados con las pruebas de varios ntree
    # 100 ->    4    6.639342  0.3610025  5.387146
                #8    6.528911  0.3748874  5.249206
                #16    6.525724  0.3684561  5.200054
    #mejor 8
    # 200 -> 
      #4    6.663659  0.3535902  5.400643
      #8    6.546280  0.3697484  5.260592
      #16    6.551002  0.3616607  5.209190
      #mejor 8
    # 400 ->
    #mtry  RMSE      Rsquared   MAE     
    #4    6.631281  0.3641442  5.384571
    #8    6.522987  0.3768531  5.244887
    #16    6.505072  0.3734240  5.184021
      #mejor 8
    # 800 -> 
      #mtry  RMSE      Rsquared   MAE     
    #4    6.622991  0.3666323  5.381355
    #8    6.524852  0.3764771  5.246091
    #16    6.504361  0.3737626  5.183834
    #mejor8
    # 1500 ->
    #mtry  RMSE      Rsquared   MAE     
    #4    6.619572  0.3677376  5.381535
    #8    6.519082  0.3781155  5.243504
    #16    6.503787  0.3740564  5.185928
    #mejor 8
    
    #nos quedamos con la opcion de 500 arboles que es casi igual que 1500 y mucho mas rapido de construir
  mtry= c(floor(sqrt(ncol(datos.train))), ncol(datos.train), floor(ncol(datos.train)/2))
  
  fitrfMotor<- train(x = datos.train, y = UPDRSmotor.train, 
                    method = "rf",
                    trControl = control,
                    tuneGrid = expand.grid(.mtry=sqrt(ncol(datos.train)) ),
                    ntree = c(500),
                    metric = "Rsquared"
  )
  fitrfTotal<- train(x = datos.train, y = UPDRStotal.train, 
                     method = "rf",
                     trControl = control,
                     tuneGrid = expand.grid(.mtry=sqrt(ncol(datos.train)) ),
                     ntree = c(500),
                     metric = "Rsquared"
  )
  
  print("Los mejores resultados obtenidos han sido para ntree= 500 y m = raiz de p")
  print("resultados para UPDRSmotor")
  print(fitrfMotor$result)
  
  print("resultados para UPDRStotal")
  print(fitrfTotal$result)
  scan(n=1)

}

##svm con nugleo RBF-Gaussiano
if(T){
    ##estas son las anotaciones de pruebas condistintos hyperparametros
    
    ## lo mejor con 1 y 0.1 da 0.3 con sigma 0.5
    #polyGrid <- expand.grid(.sigma = seq(0.45,0.6,by=0.01),
                            #.C = 2^(-2:5))
    #0.41    4.00  6.465076  0.3827206  4.894981
    #polyGrid <- expand.grid(.sigma = seq(0.4,0.45,by=0.01),
                                #.C = 2^(-2:5))
    
    
    #4.0  6.465076  0.3827206  4.894981
    #polyGrid <- expand.grid(.sigma = 0.41,
                            #.C = seq(4,6,0.5))
    #polyGrid <- expand.grid(.sigma = seq(0.30,0.35,0.01),
    #.C = seq(4.0,4.5,0.1))
    
    # mejor configuracion es 0,41 y 4 de coste 
  fitsvmRMotor = train(x = datos.train, y = UPDRSmotor.train, 
                 method = "svmRadial",
                 trControl = control,
                 #tuneGrid = polyGrid,
                 tuneGrid = expand.grid(.sigma = 0.41, .C = 4),
                 metric = "Rsquared"
                 
  )
  
  fitsvmRTotal = train(x = datos.train, y = UPDRStotal.train, 
                  method = "svmRadial",
                  trControl = control,
                  #tuneGrid = polyGrid,
                  tuneGrid = expand.grid(.sigma = 0.34, .C = 4.5),
                  metric = "Rsquared"
                  
  )
  
  print("Los mejores resultados obtenidos han sido para sigma = 0.41 y coste =4 ")
  print("resultados para UPDRSmotor")
  print(fitsvmRMotor$result)
  
  print("Los mejores resultados obtenidos han sido para sigma = 0.34 y coste =4.5 ")
  print("resultados para UPDRStotal")
  print(fitsvmRTotal$result)
  scan(n=1)


}

if(F){
  #No hemos encontrado ningun hiperparametro que mejore el nucleo anterior por lo que lo
  #descartamos para este problema
  polyGrid <- expand.grid(degree = 2,
                          scale = c(0.1),
                          C = c(2))
  
  fitsvmPMotor = train(x = datos.train, y = UPDRSmotor.train, 
                 method = "svmPoly",
                 trControl = control,
                 metric = "Rsquared",
                 tuneGrid = polyGrid
                 
  )
  fitsvmPTotal = train(x = datos.train, y = UPDRStotal.train, 
                       method = "svmPoly",
                       trControl = control,
                       metric = "Rsquared",
                       tuneGrid = polyGrid
                       
  )
  print("Los mejores resultados obtenidos han sido para degree")
  print("resultados para UPDRSmotor")
  print(fitsvmPMotor$result)
  
  print("Los mejores resultados obtenidos han sido para degree ")
  print("resultados para UPDRStotal")
  print(fitsvmPTotal$result)
  scan(n=1)

}

##boosting
if(T){
  ##boosting supuestamente probado de 2 a 100 las posibles combinaciones entre los dos hiperparametros
  #lo mejor es 51 arbol profundidad 8 para motor
  #para total usamos 55 y 9
  fitboostMotor <- train(x = datos.train, y = UPDRSmotor.train, 
                    method = "blackboost", 
                    trControl = control,
                    metric = "Rsquared",
                    tuneGrid = expand.grid(.mstop = 51, .maxdepth =8)
                    
  )
  fitboostTotal <- train(x = datos.train, y = UPDRStotal.train, 
                         method = "blackboost", 
                         trControl = control,
                         metric = "Rsquared",
                         tuneGrid = expand.grid(.mstop = 55, .maxdepth =9)
                         
  )
  print("Los mejores resultados obtenidos han sido para número de árboles =51 t profundidad =8")
  print("resultados para UPDRSmotor")
  print(fitboostMotor$result)
  
  print("Los mejores resultados obtenidos han sido para número de árboles =55 t profundidad =9")
  print("resultados para UPDRStotal")
  print(fitboostTotal$result)
  scan(n=1)
}

##seleccion del modelo final

#resultados finales para UPDRSmotor

yhat = predict(fitrfMotor,datos.test)
randomforest = postResample(yhat,UPDRSmotor.test)

yhat = predict(fitboostMotor,datos.test)
boosting = postResample(yhat,UPDRSmotor.test)

yhat = predict(fitsvmRMotor,datos.test)
svmR = postResample(yhat,UPDRSmotor.test)

Emotor = rbind(randomforest,boosting,svmR)

#resultados finales para UPDRStotal

yhat = predict(fitrfTotal,datos.test)
randomforest = postResample(yhat,UPDRStotal.test)

yhat = predict(fitboostTotal,datos.test)
boosting = postResample(yhat,UPDRStotal.test)

yhat = predict(fitsvmRTotal,datos.test)
svmR = postResample(yhat,UPDRStotal.test)

Etotal = rbind(randomforest,boosting,svmR)

#imprimimos las tablas de resultados
print("tabla de resultados finales para UPDRSmotor")
print(Emotor)
print("tabla de resultados finales para UPDRStotal")
print(Etotal)



#[1] "tabla de resultados finales para UPDRSmotor"
#RMSE  Rsquared      MAE
#randomforest 6.486063 0.3557753 5.212249
#boosting     7.069347 0.2433868 5.873745
#svmR         6.325670 0.3786563 4.859415
#[1] "tabla de resultados finales para UPDRStotal"
#RMSE  Rsquared      MAE
#randomforest 8.451121 0.3674882 6.586904
#boosting     9.258921 0.2534054 7.423456
#svmR         8.127070 0.4018044 6.155997

