setwd("~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload")

library(ggplot2)
library(latex2exp)

linf_fit <- function(test_var2,alpha,wd,printlevel = 0,path,flist,b_list,b_exp = c(),N_exp= c(),p_dir){
  print(p_dir)
  j = 1
  l = 0
  #file_list = files[files$test_str2 == test_var2, "files"]
  #print(file_list)
  for (files in flist){
    l = l + 1
    data= read.csv(files,header = TRUE,sep=',')
    #print(data)
    data = data[data[, "ttt"] > 0.0, ]
    #data = data[data[, "ttt"] < 80.0, ]
    
    #
    
    data = data[data[, "dist"] < 2.0,]
    data = data[data[, "dist"] > 0.2, ]

    x = data$dist
    y = data$ttt
    #run = data$run
     # x = data[data$run == 0, "r"]
    #y = data[data$run == 0, "ttt"]
    
    ln_x = log(x)
    ln_y = log(y)
    log_frame = data.frame(x = x, y = y)
    b_plot = c(1.0,1.6,1.7,2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5,6.0)
    #if(b_list[l] %in% b_exp && printlevel > 0){
    #b_list[l] %in% b_plot &&
    if(printlevel > 0){
      #print(b_list[l])
     # print(b_list[l] %in% b_exp)
      p = ggplot(log_frame,aes(x, y)) + geom_point(alpha=0.1)    + theme(text = element_text(size = 12))  + ylab(TeX("$T_{t}$ in s")) + xlab(TeX("$r$ in m")) + ggtitle(toString(b_list[l])) + scale_x_log10() + scale_y_log10() #+ geom_smooth(method = "lm") 
      print(p)
      if (test_var2 == "hm"){
        test_var2 = 1/100
      }
      if (test_var2 == "lm"){
        test_var2 = 0
      }
      ggsave(paste(path,"plots/",p_dir,"waitingtimelong_b_",toString(as.integer(b_list[l]*100)),"_T_",toString(as.integer(test_var2*100)),".png",sep = ""),width = 5, height = 4)
    }
    fit <- lm(ln_y ~ ln_x)
    #print(summary(fit))
    alpha_i = coef(summary(fit))[2]
    alpha = c(alpha,alpha_i)
  }
  j=j+1
  
  return(alpha)
}

exp_b <- function(motexp,alpha,path,b_exp,N_exp,file,print_exp,test_var,p_dir = p_dir){
  files = file[file$motivation == motexp, "files"]
  test_var = c(test_var,b_exp)
  alpha = linf_fit(motexp,alpha,wd_exp,print_exp,path,files,b_exp,b_exp,N_exp,p_dir = p_dir)
  return(alpha)
}

exp_N <- function(motexp,alpha,path,b_exp,N_exp,file,print_exp,test_var,p_dir){

  files = file[file$motivation == motexp & file$b == 5.6 , "files"] # changed for N
  N_exp = file[file$motivation == motexp & file$b == 5.6,"N"] # too
  
  test_var = c(test_var,N_exp)
  alpha = linf_fit(motexp,alpha,wd_exp,print_exp,path,files,b_exp,b_exp,N_exp, p_dir)
  return_list = list("alpha" = alpha, "test_var" = test_var)
  return(return_list)
}

col_maker <- function(list){
    for (i in list){
      col = c(col,"Sim") 
    }
  return(col)
}

col_chooser <- function(variable,b_list,N_list){
  if (variable == "b"){
    col = col_maker(b_list)
  }
  if (variable == "N"){
    col = col_maker(N_list)
  }
  return(col)
}

test_var_fetch <- function(variable,files,test_var2,b_list){
  if (variable == "N"){
    test_var = files[files$test_str2 == test_var2, "test_str"]
  }
  if (variable == "b"){
    test_var =  b_list #if b is test var and format is depricated
  }
  return(test_var)
}

plot_alpha <- function(test_var2, test_str, motexp, path, exp, b_list, N_list, col, printlevel, N_exp, b_exp,variable,p_dir, print_exp = 0){
  
  wd_sim = "~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload"
  col = col_chooser(variable,b_list,N_list)
  setwd(wd_sim)
  alpha = c()
  files = read.csv(paste(path,"waittime/file_list.csv",sep=""),header = TRUE,sep=',')
  print(files)
  #print(files)
  test_var = test_var_fetch(variable,files,test_var2,b_list)
  print(test_var)
  flist = files[files$test_str2 == test_var2, "files"]
  print("flist = ", flist)
  alpha = linf_fit(test_var2,alpha,wd_sim,printlevel,path,flist,test_var,b_exp,p_dir = p_dir)
  print("alpha = " ,alpha)
 # print("alphalength = " ,length(alpha))
  if (exp == TRUE){
    wd_exp = "~/Documents/phd/python/TwoPointCharme"
    setwd(wd_exp)
    files = read.csv("waittime/file_list.csv",header = TRUE,sep = ",")
    #files = file[file$motivation == motexp,"files"]
    if(variable == "b"){
      for (i in b_exp){
        col = c(col,"Exp")
      }
      test_var = c(b_list,b_exp)
      alpha = exp_b(motexp,alpha,path,b_exp,N_exp,files,print_exp,test_var,p_dir)
    }
    if(variable == "N"){
      #test_var = c(N_list,b_exp)
      print("r")
      result = exp_N(motexp,alpha,path,b_exp,N_exp,files,print_exp,test_var,p_dir)
      print("/r")
      alpha = result$alpha
      test_var = result$test_var
      col = c(col,"Exp","Exp")
    }
  }
  print( length(test_var))
  print(length(alpha))
  print( length(col))
  
  print("test")
  alpha_frame = data.frame(test_var = test_var, alpha = alpha, col = col)
  #print(alpha_frame)
  p_sim = ggplot(alpha_frame,aes(test_var,alpha,color = col)) + geom_point() + scale_color_discrete(name = "Results") +  theme(text = element_text(size=50)) + theme_classic() + ylab(TeX("$\\alpha$")) + xlab(TeX(paste("$",variable,"$",sep = "")))
  print(p_sim)
  setwd("~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload")
  ggsave(paste(path,"plots/alpha_",motexp,".pdf",sep = ""),width = 8, height = 5)
}

wd_sim = "~/Documents/phd/c++/jps_ben_vel/jpscore/files/jureca_upload"
path=read.csv("path.csv",header = TRUE,sep=',')$path
#path = "trajectories/ini_lm_N_ped55_tmax310_size_0_17_fps_16_testvar_b/" # wedge shaped sim
#path = "trajectories/ini_lm_N_ped55_tmax310_size_0_17_fps_16_testvar_N_ped/" # testing for nped with high mot
#path = "trajectories/ini_lm_N_ped55_tmax308_size_0_17_fps_16_testvar_b/" # main sim

#path = "trajectories/ini_lm_N_ped500_tmax3000_size_0_17_fps_16_testvar_b/"
#N_list = read.csv(paste(path,"N_list.csv",sep = ""),header = TRUE, sep = ",")$N
#b_list = c(5.6)
b_list = read.csv(paste(path,"b_list.csv",sep = ""),header = TRUE, sep = ",")$b
#b_list = b_list[seq(1, length(b_list), 2)]
N_list = read.csv(paste(path,"N_ped_list.csv",sep = ""),header = TRUE, sep = ",")$N

b_list = 2 * b_list
print(b_list)
col = c()
b_exp = c(1.2,2.3,3.4,4.5,4.5,5.6,5.6)
N_exp = c(64,42,67,42,42,57,75)


#alpha_plot(1.3,"test_str","lm",path,T,b_list,N_list,col,p_level,N_exp,b_exp)
exp_dat = T
p_dir = "TTTlong13/"

p_level = 0

plot_alpha(1.3,"test_str","lm",path,exp_dat,b_list,N_list,col,p_level,N_exp,b_exp,"b",p_dir, print_exp = 0)
#p_dir = "TTTlong/"
p_level = 0

plot_alpha(0.1,"test_str","lm",path,exp_dat,b_list,N_list,col,p_level,N_exp,b_exp,"b",p_dir, print_exp = 0)

