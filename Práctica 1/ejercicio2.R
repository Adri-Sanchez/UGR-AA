#--------------------------------------------------------
#           EJERCICIO SOBRE REGRESION LINEAL
#-------------------------------------------------------


#---------------------------


setwd("./datos")  #directorio de trabajo


#leer el zip.train---------------------------------------------------------------
digit.train = read.table("zip.train",quote="\"", comment.char="", stringsAsFactors=FALSE)


digitos15.train = digit.train[digit.train$V1==1 | digit.train$V1==5,]
digitos = digitos15.train[,1]    # vector de etiquetas del train
ndigitosTr = nrow(digitos15.train)  # numero de muestras del train

# se retira la clase y se monta una matriz 3D: 599*16*16
grises = array(unlist(subset(digitos15.train,select=-V1)),c(ndigitosTr,16,16))


par(mfrow=c(2,2)) 
for(i in 1:4){
  imagen = grises[i,,16:1] # se rota para verlo bien
  image(z=imagen)
}
digitos[1:4] # etiquetas correspondientes a las 4 imágenes

par(mfrow = c(1,1))
print("Estas son un ejemplo de los digitos leidos y de los cuales solo hemos seleccionados los 1 y los 5")
scan(n=1)


rm(digit.train) 
rm(digitos15.train)

#------------------------
#apartado2

print("Extraemos de cada imagen las caracteristicas de intensidad y simetria")
intensidad = apply (grises,1,mean)

fsimetria <- function(A){
  A = abs(A-A[,ncol(A):1])
  -sum(A)
}
simetria = apply (grises,1,fsimetria)

etiquetasTr =digitos
etiquetasTr[etiquetasTr == 5 ] = -1

datosTr = as.matrix(cbind(intensidad,simetria)) 
rm(grises)

print("aqui tenemos la relacion de estas dos caracteristicas")
plot(datosTr,xlab=" intensidad",ylab="simetria", main="datos de train")  
scan(n=1)


print("vamos a pintar ahora cada punto con el color de su clasificacion")
plot(datosTr,xlab=" intensidad",ylab="simetria", main="datos de train", col = etiquetasTr+2)
scan(n=1)

print("Vamos a realizar el mismo proceso para los datos de test")
#--------------------------------------------------------------------------
#leer el zip.test---------------------------------------------------------------
digit.test = read.table("zip.test",quote="\"", comment.char="", stringsAsFactors=FALSE)





digitos15.test = digit.test[digit.test$V1==1 | digit.test$V1==5,]
digitos = digitos15.test[,1]    # vector de etiquetas del test
ndigitosTst = nrow(digitos15.test)  # numero de muestras del test

# se retira la clase y se monta una matriz 3D: 599*16*16
grises = array(unlist(subset(digitos15.test,select=-V1)),c(ndigitosTst,16,16))
grises = as.numeric(grises)
dim(grises) = c(49,16,16)

rm(digit.test) 
rm(digitos15.test)

par(mfrow=c(2,2)) 
for(i in 1:4){
 imagen = grises[i,,16:1] # se rota para verlo bien
  image(z=imagen)
}

par(mfrow = c(1,1))
print("Como podemos ver tenermos el mismo tipo de datos que en train los 1 y los 5")
scan(n=1)


intensidad = apply (grises,1,mean)

simetria = apply (grises,1,fsimetria)

etiquetasTst =digitos
etiquetasTst[etiquetasTst== 5 ] = -1

datosTst = as.matrix(cbind(intensidad,simetria)) 
rm(grises)

print("vamos a pintar ahora cada punto con el color de su clasificacion")
plot(datosTst,xlab=" intensidad",ylab="simetria", main="datos de test", col = etiquetasTst+2)
setwd("~/Documentos/aa practias")
scan(n=1)


print("ya tenemos los datos necesarios, ahora buscaremos un modelo que nos sirva de clasificador")



#------------------------------------------
#apartado3
##
pasoARecta= function(w){
  if(length(w)!= 3)
    stop("Solo tiene sentido con 3 pesos")
  a = -w[1]/w[2]
  b = -w[3]/w[2]
  c(a,b)
}

print("En primer lugar vamos a usar el algoritmo de la pseudo-inversa y veremos como nos clasifica")

#funcion que calcula la regresion lineal con la pseudoinversa
Regress_Lin = function(datos,label){
  
  datos = cbind(datos,c(rep(1,length(label))))
  descom = svd(datos)
  D = descom$d
  V = descom$v
  U = descom$u
  
  pseudo = (V %*% diag(1/D) %*% t(U))
  w = pseudo %*% label
  t(w)
}


w = Regress_Lin(datosTr,etiquetasTr)

print("la regresion lineal nos a dado los siguientes pesos, vamos a calcular su error en la muestra Ein")
print(w)
scan(n=1)

#funcion para el calculo del error
calculoE = function(w,datos, label){
  datos1 = cbind(datos,1)
  #print(dim(w))
  #print(dim(datos1))
  E = sum(sign( (datos1%*%t(w)) ) != label)/length(label)
  #print("minimos")
  E  
}

Ein = calculoE(w,datosTr,etiquetasTr)
Eout = calculoE(w,datosTst,etiquetasTst)

print("Hemos obtenido los siguientes errores")
print("Ein:")
print(Ein)
print("Eout:")
print(Eout)
print("vamos a ver como nos ha clasificado los datos de test en la grafica con esos pesos")
ab = pasoARecta(w)

plot(datosTr,xlab=" intensidad",ylab="simetria", main="datos de test", col = sign(cbind(datosTr,1)%*%t(w))+2)
abline(ab[2],ab[1])
plot(datosTst,xlab=" intensidad",ylab="simetria", main="datos de test", col = sign(cbind(datosTst,1)%*%t(w))+2)
abline(ab[2],ab[1])
scan(n=1)


print("Vamos ahora a buscar nuestro clasificador con el gradiente descendente estocastico usando mu=0.05")


##algoritmo del gradiente desccendiente estocastico
GDS =function(datos,etiquetas,w,mu, maxit,umbral,umbraldiferencia = 0.001){
  e=10000
  i=0
  anterior = 100000
  diferencia = 100000
  n=length(etiquetas)
  datos1 = cbind(datos,1)

  
  while ( i<maxit  & e > umbral & diferencia < umbraldiferencia){
    indices = sample(1:length(etiquetas),60)
    
    datosmio = datos1[indices,]
    etiquetasmio = etiquetas[indices]

    for(j in 1:length(w)){
      
      w[j] = w[j] - mu* 1/60 *sum(( -(etiquetasmio * datosmio[,j]) / (1 + exp(1)^(etiquetasmio * (datosmio) %*% t(w) )) ) )
    }
    i = i+1
    e = calculoE(w,datos,etiquetas)
    diferencia = abs(e-anterior)
    anterior = e
  }
  w
}

w = GDS(datosTr,etiquetasTr,w, 0.05,10000,0.0001)

print("En este caso hemos obtenidos los siguientes pesos w:")
print("vamos a calcular el error tanto dentro como fuera en los datos de test")
Ein = calculoE(w,datosTr,etiquetasTr)
Eout = calculoE(w,datosTst,etiquetasTst)

print("Hemos obtenido los siguientes errores")
print("Ein:")
print(Ein)
print("Eout:")
print(Eout)


print("vamos a ver como nos ha clasificado los datos de test en la grafica con esos pesos")
ab = pasoARecta(w)
plot(datosTr,xlab=" intensidad",ylab="simetria", main="datos de test", col = sign(cbind(datosTr,1)%*%t(w))+2)
abline(ab[2],ab[1])
plot(datosTst,xlab=" intensidad",ylab="simetria", main="datos de test", col = sign(cbind(datosTst,1)%*%t(w))+2)
abline(ab[2],ab[1])
scan(n=1)



#----------------------------------------------
#EXPERIMENTO apartado 4
#
#
set.seed(45)

print("Ahora vamos a experimentar como evoluciona el error segun la complejidad del modelo usado, además de añadir ruido a las muestras")
simula_unif = function (N=2,dims=2, rango = c(0,1)){
  m = matrix(runif(N*dims, min=rango[1], max=rango[2]),
             nrow = N, ncol=dims, byrow=T)
  m
}



#funcion para asignar etiquetas a cada dato
f = function(x1,x2){
  sign( (x1 - 0.2)^2 + (x2^2 -0.6))
  
}


x = simula_unif(1000,2,c(-1,1))

etiquetas = f(x[,1],x[,2])


print("Hemos generado con simula unif una serie de datos y ademas los hemos calsificado con la funcion obteniendo")
plot(x,col=etiquetas+2)
scan(n=1)

print("ahora vamos a añadirle ruido cambiando la calsificacion aleotoriamente al 10%")

genera_ruido = function(etiquetas){
  n = length(etiquetas)
  inidices = sample(1:n,round(n*0.1))
  etiquetas[inidices] = -etiquetas[inidices]
  etiquetas
}

etiquetas = genera_ruido(etiquetas)

print("Ahora tenemos la siguiente situacion")
plot(x,col= etiquetas+2)
scan(n=1)

print("Vamos a aplicar en primer lugar la regresion lineal")
w = Regress_Lin(x,etiquetas)
datosmio = cbind(x,1)
print("con la regresion lineal hemos conseguido un Ein")
Ein = calculoE(w,x,etiquetas)
print(Ein)
print("Y la grafica clasificada con esto seria")
plot(x,col = sign(datosmio%*%t(w))+2)
ab = pasoARecta(w)
abline(ab[2],ab[1])
scan(n=1)

print("Vamos a usar ahora el GDS con mu = 0.05")
w =  GDS(x, etiquetas,w,0.05,10000,0.001)
print("con GDS hemos conseguido un Ein del GDS:")
Ein = calculoE(w,x,etiquetas)
print(Ein)
print("Y la grafica clasificada con esto seria")
plot(x,col = sign(datosmio%*%t(w))+2)
ab = pasoARecta(w)
abline(ab[2],ab[1])
scan(n=1)

print("Para el Experimento vamos a realizar esto mismo 1000 ademas de calcular un error out generando una muestra aleatoria de test")
print("!!CALMA, TARDA ALREDEDOR DE UN MINUTO!!!!!")

RsumaEin =0
RsumaEout=0
GsumaEin =0
GsumaEout=0

for(i in 1:1000){
  
  #generar los datos aleatorios
  xtrain = simula_unif(1000,2,-1:1)
  etiquetasTr = f(x[,1],x[,2])
  xtest = simula_unif(1000,2,-1:1)
  etiquetasTst = f(x[,1],x[,2])
  
  #le metemos ruido a las etiquetas
  etiquetasTr = genera_ruido(etiquetasTr)
  
  
   #usamos la regresion lineal
  w = Regress_Lin(xtrain,etiquetasTr)
  Ein = calculoE(w,xtrain,etiquetasTr)
  Eout = calculoE(w,xtest,etiquetasTst)
  
  RsumaEin = (RsumaEin + Ein)
  RsumaEout = (RsumaEout + Eout)
  
  #usamos el gradiente 
  w =  GDS(x, etiquetas,w,0.01,1000,0.1)
  
  Ein = calculoE(w,xtrain,etiquetasTr)
  Eout = calculoE(w,xtest,etiquetasTst)
  
  GsumaEin = (GsumaEin + Ein)
  GsumaEout =(GsumaEout + Eout)
}

print("Los resultados de la regresion lineal")
print("Ein medio")
print(RsumaEin/1000)
print("Eout medio")
print(RsumaEout/1000)


print("Los resultados de GDS")
print("Ein medio")
print(GsumaEin/1000)
print("Eout medio")
print(GsumaEout/1000)
scan(n=1)

print("La valoracion de estos datos la veremos en el .pdf")

rm(list = ls())

















