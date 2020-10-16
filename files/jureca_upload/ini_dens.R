setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload/")
exp_data_low = read.csv("exp_results/exp_data_inilow_mot.csv")
exp_data_high = read.csv("exp_results/exp_data_inihigh_mot.csv")
b_exp = c(1.2,2.3,3.4,4.5,4.5,5.6,5.6,1.2,2.3,3.4,4.5,4.5,5.6,5.6)
motivation = c("low","low","low","low","low","low","low")
motivation = c(motivation,"high","high","high","high","high","high","high")
exp_plot = data.frame(b = b_exp,mean = c(exp_data_low$mean,exp_data_high$mean), motivation = motivation )

setwd("~/Documents/phd/c++/jpscorenew/jpscore/files/jureca_upload")
path=read.csv("path.csv",header = TRUE,sep=',')$path 
dens_folder = paste(path,"density/",sep = "")

dens_plot_ini = read.csv(paste(dens_folder,"error_plot_ini.csv",sep = ""))
scat_size = 2.5
p_ini = ggplot(dens_plot_ini,aes(b,mean, color = as.character(t))) + geom_point(size = scat_size) + scale_color_discrete(name = "T/motivation") + xlab(TeX("$b in $m$")) + ylab(TeX("$\\rho$ in $m^{-2}$"))  + 
  theme(text = element_text(size=70)) + theme_classic() +
  geom_errorbar(aes(ymin=mean-errordown, ymax=mean + errorup)) + 
  geom_point(data = exp_plot, mapping = aes(b,mean, color = motivation), size = scat_size)
print(p_ini)
ggsave(paste(path,"plots/rho_ini.pdf",sep = ""))
