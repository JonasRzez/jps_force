library(TDA)
library(FNN)
library(igraph)
library(scales)
library(rgl)
setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")

#data = read.csv("test_traj.txt",header = TRUE,sep='\t',comment = "#")
data = read.csv("hexagonal.csv",header = TRUE,sep=',',comment = "#")


#print(head(data))

#data_t = data[data[,"FR"] == 25 * 16,]
data_t = data
print(data_t)
x = data_t$X
y = data_t$Y
print(x)
plot(x,y)
xy = c(x,y)
print(xy)
nc = length(x)
X = matrix(xy,nrow = nc, ncol = 2)
print(head(X))
plot(X,  main = "Sample X",xlim = c(-5,5), ylim = c(10,15))
X <- circleUnif(n = 400, r = 1)
print(head(X))
Xlim <- c(-5, 5)
Ylim <- c(10, 20)
by <- 0.05
Xseq <- seq(from = Xlim[1], to = Xlim[2], by = by)
Yseq <- seq(from = Ylim[1], to = Ylim[2], by = by)
Grid <- expand.grid(Xseq, Yseq)
#print(head(Grid))
distance <- distFct(X = X, Grid = Grid)
persp(x = Xseq, y = Yseq,
      z = matrix(distance, nrow = length(Xseq), ncol = length(Yseq)),
      xlab = "", ylab = "", zlab = "", theta = -20, phi = 35, scale = FALSE,
      expand = 3, col = "red", border = NA, ltheta = 50, shade = 0.5,
      main = "Distance Function")
par(mfrow = c(1,2))

plot(X, xlab = "", ylab = "", main = "Sample X")

Diag <- gridDiag(X = X, FUN = distFct, lim = cbind(Xlim, Ylim), by = by,
                 sublevel = F, library = "Dionysus", printProgress = FALSE,location = F)

band <- bootstrapBand(X = X, FUN = kde, Grid = Grid, B = 100,
                      parallel = FALSE, alpha = 0.1, h = 0.6)
#print(band[["width"]])

persp(x = Xseq, y = Yseq,
      z = matrix(distance, nrow = length(Xseq), ncol = length(Yseq)),
      xlab = "", ylab = "", zlab = "", theta = -20, phi = 35, scale = FALSE,
      expand = 3, col = "red", border = NA, ltheta = 50, shade = 0.9,
      main = "distFct")

plot(x = Diag[["diagram"]],band = 2 * band[["width"]], barcode = F)

#print(Diag[["diagram"]])
dia_length = round(length(Diag[["diagram"]])/3)
#print(dia_length)
simplex = Diag[["diagram"]][1:dia_length]
#print(simplex)
birth = Diag[["diagram"]][dia_length + 1:round(2 * dia_length)]
#print(birth)

death = Diag[["diagram"]][round(2 * dia_length + 1):round(dia_length * 3)]
#print(death)

#print(birth)
df = data.frame(simp = simplex, birth = birth, death = death)
#print(df$simp)
h0 = df[df[,'simp'] == 0,]
h1 = df[df[,'simp'] == 1,]

print(h0)
dh0 = h0$death - h0$birth
dh1 = h1$death - h1$birth

print(var(dh0))
print(sum(dh1))
print("hexagonal = ")
print(2 * var(dh0) + 1/(2 - sqrt(2)) * sum(dh1))
