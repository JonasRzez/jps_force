setwd("~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload")
library(ggplot2)
library(latex2exp)

linf_fit <- function(Ti,alpha,wd,sim,b,printlevel = 0,path){
  j = 1
  setwd(wd)
  print(wd)
  
  if (sim == "sim"){
    print(path)
    files=read.csv(paste(path,"waittime/file_list.csv",sep=""),header = TRUE,sep=',')
    print(Ti)

    print("sim is true")
    file_list = files[files$test_str2 == Ti, "files"]}
  if (sim == "exp"){
    files=read.csv("waittime/file_list.csv",header = TRUE,sep=',')
    print(Ti)

    print("exp is true")
    file_list = files[files$motivation == Ti, "files"]
    print(file_list)
  }

  #print(file_list)
  l = 0
  for (files in file_list){
    b_exp = c(1.2,2.3,3.4,4.5,4.5,5.6,5.6)
    
    l = l + 1
    #print(files)
    data=read.csv(files,header = TRUE,sep=',')
    data = data[data[, "ttt"] > 0.0, ]
    data = data[data[, "ttt"] < 80.0, ]
    
    data = data[data[, "r"] < 2.0, ]
    data = data[data[, "r"] > 0.1, ]
    x = data$r
    y = data$ttt
    
    ln_x = log(x)
    ln_y = log(y)
    log_frame = data.frame(x = x, y = y)
    
    if(b[l] %in% b_exp && printlevel > 0){
      print(b[l])
      print(b[l] %in% b_exp)
      p = ggplot(log_frame,aes(x, y)) + geom_point(alpha=0.1)  + scale_x_log10() + scale_y_log10() +   theme(text = element_text(size=50)) + theme_classic() + ylab(TeX("T_{t} in sec")) + xlab(TeX("x in m")) + geom_smooth(method = "lm")
      print(p)
      ggsave(paste(path,"plots/",toString(b[l]),toString(Ti),".pdf",sep = ""))
    }
    fit <- lm(ln_y ~ ln_x)
    #print(summary(fit))
    alpha_i = coef(summary(fit))[2]
    alpha = c(alpha,alpha_i)
  }
  j=j+1
  
  return(alpha)
}

alpha_plot <- function(motsim,motexp,b_sim,b_exp,col,path){
  wd_sim = "~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload"
  wd_exp = "~/Documents/phd/python/TwoPointCharme"
  alpha = c()
  alpha = linf_fit(motsim,alpha,wd_sim,"sim",b_sim,0,path)
  alpha = linf_fit(motexp,alpha,wd_exp,"exp",b_exp,0,path)
  b = c(b_sim,b_exp)
  print( length(b))
  print( length(alpha))
  print( length(col))
  
  
  alpha_frame = data.frame(b = b, alpha = alpha,col = col)
  p_sim = ggplot(alpha_frame,aes(b,alpha,color = col)) + geom_point() + scale_color_discrete(name = "Results") +  theme(text = element_text(size=50)) + theme_classic() + ylab(TeX("$\\alpha$")) + xlab(TeX("b"))
  print(p_sim)
  setwd("~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload")
  ggsave(paste(path,"plots/alpha_",motexp,".pdf",sep = ""))
  
}

b_sim = c()
for(b_i in 11:70){
  b_sim = c(b_sim,b_i/10)
}
b_sim = c(1.2,2.3,3.4,4.5,5.6)
b_exp = c(1.2,2.3,3.4,4.5,4.5,5.6,5.6)
col = c()

for (i in b_sim){
  col = c(col,"Sim") 
}
for (i in b_exp){
  col = c(col,"Exp") 
}
path=read.csv("path.csv",header = TRUE,sep=',')$path
#path = "trajectories/ini_lm_N_ped55_tmax150_size_0_18_fps_16_testvar_b/"
#ini_lm_N_ped55_tmax303_size_0_17_fps_16_testvar_b
print(path)
#alpha_plot(0.7,"lm",b_sim,b_exp,col,path)

#alpha_plot(0.8,"lm",b_sim,b_exp,col,path)
#alpha_plot(0.9,"lm",b_sim,b_exp,col,path)

#alpha_plot(1.0,"lm",b_sim,b_exp,col,path)
alpha_plot(1.3,"lm",b_sim,b_exp,col,path)
#alpha_plot(1.2,"lm",b_sim,b_exp,col,path)
#alpha_plot(1.3,"lm",b_sim,b_exp,col,path)


#alpha_plot(0.1,"hm",b_sim,b_exp,col,path)





