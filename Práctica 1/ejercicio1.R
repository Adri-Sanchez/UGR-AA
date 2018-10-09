## Practicas de Aprendizaje Automático
## Grupo AA2
## Curso 2017/2018
## Autores: Adrián Sánchez Cerrillo y Miguel Ángel López Robles
## DNI: 76655183R

## correo: adrisanchez@correo.ugr.es | robles2197@correo.ugr.es
## ------------------------------------------------------------------------

f1.2 <- function(u,v){
  fxy <- (u^3 * (exp(1)^(v-2)) - 4*v^3 * (exp(1)^-u) )^2

  fxy
}

d1.2 <- function(u,v){
  du <- 2*(u^3 * (exp(1)^(v-2))-4*(exp(1)^(-u))*v^3) * (3*u^2*(exp(1)^(v-2))+4 * (exp(1)^(-u)) * v^3)
  dv <- 2*(u^3 * (exp(1)^(v-2))-12*(exp(1)^(-u))*v^2)* (u^3 * (exp(1)^(v-2))-4 * (exp(1)^(-u)) * v^3)
  
  duv <- c(du,dv)
  
  duv
}

f1.3 <- function(x,y){
  fxy <- (x-2)^2 + 2*(y+2)^2 + 2*sin(2*pi*x) * sin(2*pi*y)
  
  fxy
}

d1.3 <- function(x,y){
  dx <- 4*pi*cos(2*pi*x)*sin(2*pi*y)+2*(x-2) 
  dy <- 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y+2)
  
  dxy <- c(dx,dy)
  
  dxy
}

## ------------------------------------------------------------------------
##            1. Implementación del algoritmo Gradiente Descendente
## ------------------------------------------------------------------------

GD <- function(f, df, wini = c(0,0), nitr = 100000, lr = 0.1, umbral = 1*10^(-10), umbraldif = 1*10^(-10)) {
  iteraciones <- 0
  min <- .Machine$integer.max
  diferencia <- .Machine$integer.max
  dif_significativa <- TRUE
  
  valores <- f(wini[1], wini[2])
  
  while(iteraciones < nitr & min > umbral & dif_significativa ){
    anterior <- f(wini[1], wini[2])
    
    wini <- wini - lr * df(wini[1],wini[2])
    
    min <- f(wini[1],wini[2]) 
    valores <- append(valores, min)
    diferencia <- abs(anterior-min)
    
    dif_significativa <- diferencia > umbraldif
    
    iteraciones = iteraciones + 1
  }
  
  if (iteraciones >= nitr){
    print ("GD ha llegado al limite de iteraciones")
  }
  else if (min < umbral){
    print ("Se ha alcanzado un valor en la función por debajo del umbral")
  }
  else{
    print ("GD ha parado por encontrarse en una llanura de la función")
  }
  
  wini <- list(wini, iteraciones, valores)
  names(wini) <- c("w", "iteraciones", "valores")
  
  wini
}

## ------------------------------------------------------------------------
##                             2.  Función E(u,v)
## ------------------------------------------------------------------------
# B) Iteraciones para encontrar un valor inferior a 10^-14
# C) Coordenadas x,y dónde se alcanzó dicho valor
# -------------------------------------------------------------------------

# Establecemos el nº de iteraciones a un valor alto, nuestro umbral de parada a 1*10^(-13) suficiente para encontrar un valor menor al indicado.
# El valor devuelto por GD es un vector de 3 componentes, w[1] y w[2] para los pesos y w[3] para las iteraciones

w <- GD(f1.2, d1.2, wini = c(1,1), lr = 0.05, nitr = 1*10^30, umbral = 1*10^(-13), umbraldif = 1*10^(-20) )

cat ("\n", "- EJERCICIO 1.2 -", "\n")
cat ("Nº de Iteraciones en encontrar un valor inferior a 10^-14: ", w$iteraciones, "\n")
cat ("Coordenadas [x,y] dónde se encontró el valor [",w$w[1],"] [",w$w[2],"]")
scan(n=1)

## ------------------------------------------------------------------------
##                             3.  Función F(x,y)
## ------------------------------------------------------------------------
# A) Minimizar f(x,y) y realizar gráficos
# -------------------------------------------------------------------------

w1 <- GD(f1.3, d1.3, wini = c(1,1), lr = 0.01, nitr = 50, umbraldif = -1)

plot(w1$valores, type = "o")

cat ("\n", "- EJERCICIO 1.3 a) -", "\n")
cat ("Se puede observar como el GD al tener un learning rate de 0.01, se estanca en un mínimo.")
cat ("\n", "Ahora realizamos el mismo experimento pero con learning rate de 0.1")
scan(n=1)

w2 <- GD(f1.3, d1.3, wini = c(1,1), lr = 0.1, nitr = 50, umbral = -5, umbraldif = -1)

plot(w2$valores, type = "o")

scan(n=1)

cat ("\nComparación de ambas tasas de aprendizaje")

plot(w1$valores, type = "o", col = "blue", ylim=c(-2,20), ann = FALSE)
lines(w2$valores, type="o", pch=22, lty=2, col="red")
title(ylab = "f(x,y)")
title(xlab = "Iteraciones")
lines(w2$valores, type="o", pch=22, lty=2, col="red")
legend(40, 20, c("Lr 0.1", "Lr 0.01"), cex=0.8, col=c("blue","red"), pch=21:22, lty=1:2)

scan(n=1)

rm(w)
rm(w1)
rm(w2)

## ------------------------------------------------------------------------
# B) Obtener valores mínimos y realizar tabla
# -------------------------------------------------------------------------
cat ("\n", "- EJERCICIO 1.3 b) -", "\n")

ej1.3b <- matrix(c(2.1, -2.1, 3, -3, 1.5, 1.5, 1, -1), byrow = TRUE, ncol = 2)
vec <- c()

for(i in 1:nrow(ej1.3b)){
  cat ("Punto de inicio: (",ej1.3b[i, 1],", ",ej1.3b[i, 2],")", "\n")
  w <- GD(f1.3, d1.3, wini = c(ej1.3b[i, 1],ej1.3b[i, 2]), lr = 0.05, nitr = 100000, umbral = -1.5, umbraldif = 1*10^(-4) )
  
  cat ("Mínimo encontrado: ", f1.3(w$w[1], w$w[2]), "\n")
  cat ("Coordenadas [x,y] dónde se encontró el valor [",w$w[1],",",w$w[2],"]", "\n")
  cat ("Iteraciones para encontrar el mínimo: ", w$iteraciones, "\n")
  
  vec <- c(vec, w$w[1], w$w[2], f1.3(w$w[1], w$w[2]))
  
  scan(n=1)
}

cat ("\n", "Resultados obtenidos:", "\n")

tabla <- matrix(vec, byrow = TRUE, ncol = 3)
colnames(tabla) <- c("X", "Y", "Minimo")
rownames(tabla) <- c("[2.1, -2.1]", "    [3, -3]", " [1.5, 1.5]", "    [1, -1]")

print(tabla)

rm(list = ls())






