dens <- function(x){
  return(sqrt(1/(0.9069 * 3.1415 * x)))
}
rho = c()
r = c()
for(x in 8:12){
  r = c(r,x)
  rho = c(rho,dens(x))
}
print(r)
print(rho)
df = data.frame(x = r, rho = rho)

p_sim = ggplot(df,aes(r,rho)) + geom_line() +  theme(text = element_text(size=50)) + theme_classic() + xlab(TeX("$\\rho$ in $m^{-2}$")) + ylab(TeX("r in m"))
ggsave("packing_desn.pdf")
print(p_sim)