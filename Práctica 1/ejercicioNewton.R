## Practicas de Aprendizaje Automático
## Grupo AA2
## Curso 2017/2018
## Autores: Adrián Sánchez Cerrillo y Miguel Ángel López Robles
## DNI: 76655183R

## correo: adrisanchez@correo.ugr.es | robles2197@correo.ugr.es
## ------------------------------------------------------------------------

# Lo primero que tenemos que hacer es definir de modo claro la función y 
# sus derivadas necesarias

fun <- function(x, y){
  ((x-2)^2 + 2*(y+2)^2 + 2*sin(2*pi*x)*sin(2*pi*y))
}

df <- function(x, y){
  dx <- 4*pi*cos(2*pi*x)*sin(2*pi*y)+2*(x-2) 
  dy <- 4*pi*sin(2*pi*x)*cos(2*pi*y)+4*(y+2)
  
  dxy <- c(dx,dy)
  
  dxy
}

dxy <- function(x, y){
  (8*pi^2*cos(2*pi*x)*cos(2*pi*y))
}

d2x <- function(x, y){
  (2-8*pi^2*sin(2*pi*x)*sin(2*pi*y))
}


d2y <- function(x, y){
  (4-8*pi^2*sin(2*pi*x)*sin(2*pi*y))
}

dyx <- function(x, y){
  (8*pi^2*cos(2*pi*x)*cos(2*pi*y))
}

## ------------------------------------------------------------------------
##                 Implementación del Método de Newton
## ------------------------------------------------------------------------
Metodo_Newton <- function(w, f, df, d2x, d2y, dxy, dyx, mu = 0.1, 
                              umbral = 10^(-4), max_iter = 200) {

  iter = 0
  seguir = T
  anterior = .Machine$integer.max
  valores=list()
  iteraciones=list()
  
  while(iter < max_iter & seguir){
    # Calculamos la matriz Hessiana
    m <- matrix(c(d2x(w[1],w[2]), dxy(w[1],w[2]), dyx(w[1],w[2]), d2y(w[1],w[2])), ncol = 2, nrow = 2, byrow = T)
    
    gradiente = df(w[1], w[2])
    
    w = w - mu*(solve(m)%*%gradiente)
    
    valores = c(valores, f(w[1], w[2]))
    iteraciones = c(iteraciones, iter)
    
    iter = iter + 1
    if(abs(f(w[1], w[2])-anterior) < umbral){
      seguir = F
    }
    anterior = f(w[1], w[2])
  }
  
  if (iter >= max_iter){
    print ("Método de Newton ha llegado al limite de iteraciones")
  }
  else{
    print ("Método de Newton ha parado por encontrarse en una llanura de la función")
  }
  
  
  valores=unlist(valores)
  iteraciones=unlist(iteraciones)
  
  MN <- list(w, iteraciones, valores)
  names(MN) <- c("w", "iteraciones", "valores")
  
  MN
}

## ------------------------------------------------------------------------
##            Implementación del algoritmo Gradiente Descendente
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

Puntos <- matrix(c(2.1, -2.1, 3, -3, 1.5, 1.5, 1, -1), byrow = TRUE, ncol = 2)

# Forzamos que ambos métodos paren tras finalizar las iteraciones

par(mfrow=c(2,1))

for(i in 1:nrow(Puntos)){
  MNewton <- Metodo_Newton(w = c(Puntos[i,1], Puntos[i,2]), f = fun, df = df, d2x = d2x, d2y = d2y, dxy = 
                       dxy, dyx = dyx, mu = 0.1, umbral = -10^(-5), max_iter = 50 )
  
  GDescendente <- GD(fun, df, wini = c(Puntos[i,1], Puntos[i,2]), lr = 0.1, nitr = 50, umbral = -5, umbraldif = -1)
  
  plot(MNewton$valores, type = "o", col = "blue", main = "Método de Newton", xlab = "Iteraciones", ylab = "f(x,y)")
  plot(GDescendente$valores, type = "o", col = "red", main = "Gradiente Descendente", xlab = "Iteraciones", ylab = "f(x,y)")
  
  scan(n=1)
}

par(mfrow=c(1,1))

rm(list=ls())


