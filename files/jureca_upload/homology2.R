library(TDA)
library(FNN)
library(igraph)
library(scales)
library(rgl)
setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")
data_list = c("hexagonal/hexagonal0_0","hexagonal/hexagonal0_1","hexagonal/hexagonal0_2","hexagonal/hexagonal0_3","hexagonal/hexagonal0_4","hexagonal/hexagonal0_5","hexagonal/hexagonal0_6","hexagonal/hexagonal0_7","hexagonal/hexagonal0_8","hexagonal/hexagonal0_9")

#data_sim = read.csv("trajectories/ini_lm_N_ped10000_tmax100_size_0_17_fps_16_testvar_b/ini_28_5_lm_10000_esigma_0_7_tmax_100_periodic_0_v0_1_34_T_1_0_rho_ini_3_0_Nped_10000_0_motfrac_1_0/new_evac_traj_57_0_0.txt",header = TRUE,sep='\t',comment = "#")
filedata = read.csv("hexagonal/files.csv")
filelist = filedata$files
cphmean = c()
sumh1mean = c()
varh0mean = c()
mu_list = c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

for (data_name in data_list){
  cph = c()
  sumh1 = c()
  varh0 = c()
  for (i in 0:9){
    print(data_name)
    data_name_comb = paste(data_name,i,".csv",sep = "")
    print(data_name_comb)
    #data_name = data_list[1]
    print("data load")
    data = read.csv(data_name_comb,header = TRUE,sep=',',comment = "#")
    print("/data load")
    
    #print(head(data))
    #data_t = data[data[,"FR"] == 25 * 16,]
    data_t = data
    #print(data_t)
    print("dataplot")
    N_test = nrow(data_t)
    print(N_test)
    x = data_t$X
    y = data_t$Y
    #print(x)
    plot(x,y)
    xy = c(x,y)
    print("/dataplot")
    
    #print(xy)
    print("matrix")
    nc = length(x)
    X = matrix(xy,nrow = nc, ncol = 2)
    print(head(X))
    print("/matrix")
    
    #plot(X,  main = "Sample X",xlim = c(-5,5), ylim = c(10,15))
    #X <- circleUnif(n = 400, r = 1)
    print("homology")
    
    Diag <- ripsDiag(X = X, maxdimension = 1, maxscale = 2.5,
                         library = c("Dionysus"), location = TRUE,dist="euclidean")
    plot(x = Diag[["diagram"]], main = "Rips Diagram")
    plot(Diag[["diagram"]], barcode = TRUE, main = "Barcode")
    print("/homology")
    
    #print(Diag[["diagram"]])
    dia_length = round(length(Diag[["diagram"]])/3)
    #print(dia_length)
    simplex = Diag[["diagram"]][1:dia_length]
    #print(simplex)
    birth = Diag[["diagram"]][dia_length + 1:round(dia_length)]
    #print(birth)
    
    death = Diag[["diagram"]][round(2 * dia_length + 1):round(dia_length * 3)]
    #print(death)
    
    #print(birth)
    df = data.frame(simp = simplex, birth = birth, death = death)
    #print(df$simp)
    h0 = df[df[,'simp'] == 0,]
    h1 = df[df[,'simp'] == 1,]
    #print(h1)
    h0['diff'] <- (h0$death - h0$birth)
    h0 = h0[-c(1),]
    #print(var(h0$diff))
    #print(h0)
    #print(h0$diff)
    h1['diff'] = (h1$death - h1$birth)
    #hist(h0$diff)
    #print(h1$diff)
  
    
    #print(h0_fil)
    #print("var")
    #print(var(h0_fil$diff))
    #print(sum(h1$diff))
    print("hexagonal = ")
    brav_measure = var(h0$diff) + 1/(2 - sqrt(2)) * sum(h1$diff)/N_test
    print(brav_measure)
    cph = c(cph,brav_measure)
    sumh1 = c(sumh1,sum(h1$diff))
    varh0 = c(varh0,var(h0$diff))
   # hist(h0_fil$diff)
    
  }
  sumh1mean = c(sumh1mean,mean(sumh1))
  varh0mean = c(varh0mean,mean(varh0))
  
  cphmean = c(cphmean,mean(cph))
}
#print(h0_fil)
#data = read.csv("new_evac_traj_57_0_0.txt",header = TRUE,sep='\t',comment = "#")

brav_time_list = c()
h0_list = c()
h1_list = c()
time_list = c(0,1,2,5,8,10,15,20,25,30,35,40,45,50,55,90)
for (time in time_list){
  data_t = data_sim[data_sim[,"FR"] == time * 16,]
  data_t = data_t[data_t[,'X'] < 6,]
  data_t = data_t[data_t[,'X'] > -6,]
  data_t = data_t[data_t[,'Y'] > 1,]
  data_t = data_t[data_t[,'Y'] < 13,]
  N = nrow(data_t)
  
  x = data_t$X
  y = data_t$Y
  plot(x,y)
  xy = c(x,y)
  nc = length(x)
  X = matrix(xy,nrow = nc, ncol = 2)
  Diag <- ripsDiag(X = X, maxdimension = 1, maxscale = 1.,
                   library = c("Dionysus"), location = TRUE,dist="euclidean")
  plot(x = Diag[["diagram"]], main = "Rips Diagram")
  plot(Diag[["diagram"]], barcode = TRUE, main = "Barcode")
  dia_length = round(length(Diag[["diagram"]])/3)
  simplex = Diag[["diagram"]][1:dia_length]
  birth = Diag[["diagram"]][dia_length + 1:round(dia_length)]
  death = Diag[["diagram"]][round(2 * dia_length + 1):round(dia_length * 3)]
  df = data.frame(simp = simplex, birth = birth, death = death)
  h0 = df[df[,'simp'] == 0,]
  h1 = df[df[,'simp'] == 1,]
  h0['diff'] <- (h0$death - h0$birth)
  h0 = h0[-c(1),]
  h1['diff'] = (h1$death - h1$birth)
  #print(h1$diff)
  #print(h0_fil)
  #print("var")
  #print(var(h0_fil$diff))
  print(sum(h1$diff))
  print("hexagonal simulation = ")
  brav_measure = var(h0$diff) + 1/(2 - sqrt(2)) * sum(h1$diff)/N
  h0_list = c(h0_list,var(h0$diff))
  h1_list = c(h1_list,var(h0$diff))
  
  print(brav_measure)
  brav_time_list = c(brav_time_list,brav_measure)}
print(varh0)
plot(mu_list,varh0mean)
plot(mu_list,sumh1mean)
plot(time_list, brav_time_list)


#print(var(h0_fil))
#plot(mu_list,sumh1)
#plot(mu_list,varh0)
plot(mu_list,cphmean)
#print(varh0)
#print(var(h0[h0[,'diff'] == 1,]['diff']))
#print(var(c(1,1,1,1,1,1,1,1,1,1,1)))
#hist(h1$diff)
