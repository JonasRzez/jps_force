library(ggplot2)
library(latex2exp)
setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")

exp_plot_data = function(exp_path_high,exp_path_low){
  print(exp_path_high)
  exp_data_low = read.csv(exp_path_low,header = TRUE,sep=',')
  exp_data_high = read.csv(exp_path_high,header = TRUE,sep=',')
  b_exp = c(1.2,2.3,3.4,4.5,4.5,5.6,5.6,1.2,2.3,3.4,4.5,4.5,5.6,5.6)
  motivation = c("high","high","high","high","high","high","high")
  motivation = c(motivation,"low","low","low","low","low","low","low")
  
  exp_plot = data.frame(b = b_exp,mean = c(exp_data_low$mean,exp_data_high$mean), motivation = motivation )
  
}

dens_plot <- function(path,file_name,plot_name,x_label,y_label,legend_label,exp_plot_path,test_var){
  exp_plot = exp_plot_data(exp_plot_path[1],exp_plot_path[2])
  
  dens_folder = paste(path,"density/",sep = "")
  dens_plot = read.csv(paste(dens_folder,file_name,sep=""),header = TRUE,sep=',')
  scat_size = 2.
  if(test_var == "rho"){
    p_sim = ggplot(dens_plot,aes(rho,mean, color = as.character( t)))}
  if(test_var == "b"){
    p_sim = ggplot(dens_plot,aes(b,mean, color = as.character( t)))}
  if(test_var == "mot_frac"){
    print(dens_plot)
    p_sim = ggplot(dens_plot,aes(mot_frac,mean, color = as.character(t)))}
  if(test_var == "N_ped"){
    print(dens_plot)
    p_sim = ggplot(dens_plot,aes(N_ped,mean, color = as.character(t)))}
  
  p_sim = p_sim + geom_point(size = scat_size) + scale_color_discrete(name = legend_label) + ylab(TeX(y_label)) + xlab(TeX(x_label))  + 
  theme_classic(base_size = 14)
  if(test_var == "b"){p_sim = p_sim + geom_errorbar(aes(ymin=mean-errordown, ymax=mean + errorup)) + geom_point(data = exp_plot, mapping = aes(x = b,y = mean, color = motivation), size = scat_size*1.5)}
  if(test_var == "mot_frac"){p_sim = p_sim + geom_errorbar(aes(ymin=mean-errordown, ymax=mean + errorup))}
  p_sim = p_sim + scale_color_manual(breaks = c("high","low", "0.1","1.3"),values=c( "#F8766D","#7CAE00", "#00BFC4", "#C77CFF"), labels = c("high mot", "low mot", "0.1 s" , "1.3 s"), name = "")
  print(p_sim)
  ggsave(paste(path,plot_name,sep = ""))
  
}

path=read.csv("path.csv",header = TRUE,sep=',')$path
print(path)
path = "trajectories/ini_lm_N_ped55_tmax130_size_0_17_fps_16_testvar_b/"

#print(exp_plot)
#exp_plot_path = c("exp_results/exp_data_10slow_mot.csv","exp_results/exp_data_10shigh_mot.csv")
#dens_plot(path,"error_plot_10.csv","plots/dens_vergleich_10sec.pdf","$\\b$ in $m$","$\\rho$ in $m^{-2}$","T/Motivation",exp_plot_path,"b")
exp_plot_path = c("exp_results/exp_data_5slow_mot.csv","exp_results/exp_data_5shigh_mot.csv")
#dens_plot(path,"error_plot_5.csv","plots/dens_vergleich_5sec.pdf","$\\b$ in $m$","$\\rho$ in $m^{-2}$","T/Motivation",exp_plot_path,"b")

#path = "trajectories/ini_lm_N_ped55_tmax305_size_0_17_fps_16_testvar_rho/"
#dens_plot(path,"error_plot.csv","plots/dens_vergleich_ini.pdf","$\\rho_i$ in $m^{-2}$","$\\rho$ in $m^{-2}$","b in m",exp_plot_path,"rho")

#exp_plot_path = c("exp_results/exp_data_5slow_mot.csv","exp_results/exp_data_5shigh_mot.csv")
dens_plot(path,"error_plot.csv","plots/dens_vergleich.pdf","$\\b$ in $m$","$\\rho$ in $m^{-2}$","",exp_plot_path,"b")

