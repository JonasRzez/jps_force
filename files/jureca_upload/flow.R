library(ggplot2)
library(latex2exp)
setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")
path=read.csv("path.csv",header = TRUE,sep=',')$path
path = "trajectories/ini_lm_N_ped55_tmax130_size_0_17_fps_16_testvar_b/"
exp_data = read.csv("exp_results/DT.csv",header = TRUE, sep = ",")
sim_data = read.csv(paste(path,"flow/flow_err.csv",sep = ""),header = TRUE, sep = ",")
DT_data_01 = read.csv(paste(path,"flow/DThist0.1.csv",sep = ""),header = TRUE, sep = ",")
DT_data_13 = read.csv(paste(path,"flow/DThist1.3.csv",sep = ""),header = TRUE, sep = ",")
DT_exp = read.csv("exp_results/DThist.csv",header = TRUE, sep = ",")
print(head(DT_data_01))


p_sim =  ggplot(sim_data,aes(b,DT, color = as.character(T))) + geom_point(size = 4) + scale_color_discrete(name = "Mot/T[s]")  + 
  geom_errorbar(aes(ymin=DT-DTerr_down, ymax=DT + DTerr_up)) +  theme(text = element_text(size=70)) + theme_classic(base_size = 25) + geom_point(data = exp_data, mapping = aes(x = b,y = DT, color = mot), size = 4) +
  ylab(TeX("$\\Delta T$ in s")) + xlab(TeX("$b$ in m")) + scale_color_manual(labels = c("high mot", "low mot", "0.1 s" , "1.3 s") , breaks = c("high", "low", "0.1","1.3"),values=c("#F8766D","#7CAE00","#00BFC4", "#C77CFF"),name = "")
ggsave(paste(path,"plots/flow.pdf", sep = ""))
print(p_sim )

print(ggplot(DT_data_01,aes(x = DT)) + geom_histogram(bins = 80) + scale_y_log10())#+ geom_histogram(data = DT_data_13, aes(x = DT), bins = 80))
print(ggplot(DT_data_13,aes(x = DT)) + geom_histogram(bins = 80) + scale_y_log10())
print(ggplot(DT_exp,aes(x = DT, color = motivation)) + geom_histogram(aes(y = stat(count / sum(count))),bins = 30) + scale_y_log10()) 
print(ggplot(DT_exp, aes(x=DT,color = motivation)) + geom_histogram(aes(y = stat(count / sum(count))),bins = 80)  + theme_classic( base_size = 25))  

hist(DT_data_01$DT)
hist(DT_data_13$DT)
