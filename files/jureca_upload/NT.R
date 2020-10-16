library(ggplot2)
library(latex2exp)

setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")
path=read.csv("path.csv",header = TRUE,sep=',')$path
path = "trajectories/ini_lm_N_ped55_tmax129_size_0_17_fps_16_testvar_b/"
exp_data = read.csv("exp_results/NT.csv",header = TRUE, sep = ",")
sim_data = read.csv(paste(path,"flow/NT.csv",sep = ""),header = TRUE, sep = ",")
motivation = sim_data$motivation
print(motivation)

#motivation = motivation[motivation == 0.1] = "0.1 s"
#motivation = motivation[motivation == 1.3] = "1.3 s"
#sim_data$motivation = motivation

#print(sim_data)
p_sim =  ggplot(sim_data,aes(t,NT, color = as.character(motivation), group = num))  + geom_line(size = 1, alpha = 0.6) +
  geom_line(data = exp_data, mapping = aes(x = t,y = NT, color = as.character(motivation), group = num), size = 1,alpha = 1) + scale_color_manual(labels = c("high mot", "low mot", "0.1 s" , "1.3 s") , breaks = c("high", "low", "0.1","1.3"),values=c("#F8766D","#7CAE00","#00BFC4", "#C77CFF"),name = "") +
  ylab(TeX("N(t)")) + xlab(TeX("t in s")) +  theme_classic(base_size = 25)   + stat_summary(data = exp_data, fun=mean,geom="line",lwd=2,aes(group=1))
ggsave(paste(path,"plots/NT.pdf", sep = ""))
print(p_sim)

