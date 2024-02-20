# Model and parameter recovery analyses
#Alexandra Witt (2021)
library(dplyr)
library(ggplot2)
library(viridis)
library(reshape2)

source('statisticalTests.R')
addSmallLegend <- function(myPlot, pointSize = 0.5, textSize = 3, spaceLegend = 0.1) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           fil = guide_legend(override.aes = list(size = pointSize))) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          legend.key.size = unit(spaceLegend, "lines"))
}

setwd("C:/PhD/Code/socialGeneralization/Data/recovery_data/mrecov")
data <- read.csv("mrecov.csv")

data$model=factor(data$model,levels=c("AS","DB","VS","SG"))
data[,5:8] <- t(sapply(1:dim(data)[1], function(x) ifelse(data[x,5:8]==min(data[x,5:8]),1,0)))

#confusion and inversion matrix
mat <- data %>% group_by(model) %>% dplyr::summarise(AS=mean(AS_fit),DB=mean(DB_fit),VS=mean(VS_fit),SG=mean(SG_fit)) #AS=mean(AS_fit),
plotdata <- melt(as.data.frame(mat),id="model")
plotdata$model <- factor(plotdata$model,levels=c("AS","DB","VS","SG"))
plotdata$variable <- factor(plotdata$variable,levels=c("AS","DB","VS","SG"))

(conf <- ggplot(plotdata, aes(x=variable,y=reorder(model, desc(model)),fill=value))+
  geom_tile()+
  xlab("fit model")+
  ylab("generating model")+
  geom_text(aes(label = round(value, 2),color = value>0.3))+
  scale_color_manual(guide = "none", values = c("white", "black"))+
  scale_fill_viridis(name="p(fit|gen)")+ #,limits=c(0,1)
  theme_classic()+
  theme(axis.text.x = element_text(angle = 45,hjust = 1),aspect.ratio = 1))

inv_mat <- function(data, conf_mat){
  p_sim <- c(prop.table(table(data$model)))
  p_fit <- c(mean(data$AS_fit),mean(data$DB_fit),mean(data$VS_fit),mean(data$SG_fit)) #mean(data$AS_fit),
  names(p_fit) <- c("AS","DB","VS","SG")#"AS",
  inv <- apply(conf_mat,1, function(row){as.numeric(row[3])*p_sim[row[1]]/p_fit[row[2]]})
  inv_mat <- conf_mat
  inv_mat$value <- inv
  return(inv_mat)
}

inv <- inv_mat(data,plotdata)

(invmat <- ggplot(inv, aes(x=variable,y=reorder(model, desc(model)),fill=value))+
  geom_tile()+
  xlab("fit model")+
  ylab("generating model")+
  geom_text(aes(label = round(value, 2),color = value>0.4))+
  scale_color_manual(guide = "none", values = c("white", "black"))+
  scale_fill_viridis(name="p(gen|fit)")+ #,limits=c(0,1)
  theme_classic()+
  theme(axis.text.x = element_text(angle = 45,hjust = 1),aspect.ratio = 1))

mrecov <- cowplot::plot_grid(conf,invmat,nrow=1,labels="auto")
ggsave("./plots/m_recovery.pdf",height=5,width=7.5)

#parameter recovery

data <- read.csv("precov.csv")

data$model=factor(data$model,levels=c("AS","DB","VS","SG"))
data$par <- data$alpha+data$gamma+data$eps_soc
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9",  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

(pLambda <- ggplot(data,aes(x=lambda,y=lambda_fit,color=model))+
    geom_point(alpha = 0.7)+
    geom_abline(slope=1)+
    facet_wrap(~model,scales = "free", nrow=1)+
    scale_color_manual(values=cbPalette)+
    scale_y_log10()+
    scale_x_log10()+
    xlab(expression(paste("log(", lambda, ")")))+
    ylab(expression(paste("log(",lambda[italic(fit)],")")))+ 
    theme_classic()+
    theme(legend.position='none', strip.background = element_blank()))

lambda_corr <- corTestPretty(data$lambda,data$lambda_fit, method = 'kendall')

(pBeta <- ggplot(data,aes(x=beta,y=beta_fit,color=model))+
    geom_point(alpha = 0.5)+
    geom_abline(slope=1)+
    facet_wrap(~model,scales = "free",nrow = 1)+
    scale_color_manual(values=cbPalette)+
    scale_y_log10()+
    scale_x_log10()+
    xlab(expression(paste("log(", beta, ")")))+
    ylab(expression(paste("log(",beta[italic(fit)],")")))+ 
    theme_classic()+
    theme(legend.position='none', strip.background = element_blank()))

beta_corr <- corTestPretty(data$beta,data$beta_fit, method = 'kendall')

(pTau <- ggplot(data,aes(x=tau,y=tau_fit,color=model))+
    geom_point(alpha = 0.5)+
    geom_abline(slope=1)+
    facet_wrap(~model,scales = "free",nrow=1)+
    scale_color_manual(values=cbPalette)+
    scale_y_log10()+
    scale_x_log10()+
    xlab(expression(paste("log(", tau, ")")))+
    ylab(expression(paste("log(",tau[italic(fit)],")")))+ 
    theme_classic()+
    theme(legend.position='none', strip.background = element_blank()))

tau_corr <- corTestPretty(data$tau,data$tau_fit, method = 'kendall')

(parplot <- ggplot(data,aes(x=par,y=par_fit,color=model))+
    geom_point(alpha = 0.5)+
    geom_abline(slope=1)+
    facet_wrap(~model,scales = "free",nrow=1)+
    scale_color_manual(values=cbPalette)+
    scale_y_log10()+
    scale_x_log10()+
    xlab(expression(log(par[italic(soc)])))+
    ylab(expression(log(par[italic(soc[italic(fit)])])))+ 
    theme_classic()+
    theme(legend.position='none', strip.background = element_blank()))

#scale the parameters separately to avoid spurious correlations
soc_scale <- c(scale(subset(data,model=="DB")$gamma),scale(subset(data,model=="VS")$alpha),scale(subset(data,model=="SG")$eps_soc))
parfit_scale <- c(scale(subset(data,model=="DB")$par_fit),scale(subset(data,model=="VS")$par_fit),scale(subset(data,model=="SG")$par_fit))

soc_cor <- corTestPretty(soc_scale,parfit_scale,method="kendall")

gamma_cor <- corTestPretty(subset(data,model=="DB")$gamma,subset(data,model=="DB")$par_fit,  method = 'kendall')
alpha_cor <- corTestPretty(subset(data,model=="VS")$alpha,subset(data,model=="VS")$par_fit,  method = 'kendall')
eps_cor <- corTestPretty(subset(data,model=="SG")$eps_soc,subset(data,model=="SG")$par_fit,  method = 'kendall')

(parplot <- cowplot::plot_grid(pLambda, pBeta, pTau, parplot,nrow=4))
ggsave("C:/PhD/Code/socialGeneralization/plots/precovery.pdf",width=10,height=7.5)


#bounding

setwd("C:/PhD/Code/socialGeneralization")
#DB - 
sim_data = read.csv("./Data/GP_het_400_regressable.csv") 
#ignore random trials
sim_data <- subset(sim_data,trial!=0)
#get imitation stats for AS
AS_imitation <- subset(sim_data, social==1&model=="AS")%>%group_by(agent,group,round,trial)%>%summarize(imitation = any(soc_sd==0,na.rm=T))
AS_imitation <- subset(sim_data, social==1&model=="AS")%>%group_by(agent,group,round,trial)%>%summarize(imitation = any(soc_sd==0,na.rm=T))%>%group_by(agent,group,round)%>%summarize(avg_imi = mean(imitation))
AS_imitation_rate <- mean(AS_imitation$avg_imi)
AS_imitation_num <- AS_imitation_rate*14 #To get decidedly more than AS (i.e. coincidental cases due to environment structure),

#exact poisson method
# Define the desired probability (1 - 0.05 for the upper 5%)
probability <- 0.05
# Use qpois to find the threshold value
upper_threshold <- qpois(1 - probability, AS_imitation_num) 
upper_threshold_rate <- upper_threshold/14
countrange = 0:14
#show distribution in AS
pois_dist <- dpois(countrange,AS_imitation_num)
AS_imitation_dist <- data.frame(AS_imitation = countrange, Probability = pois_dist)

(DB_just <- ggplot(AS_imitation_dist,aes(x=AS_imitation/14,y=Probability))+
  geom_bar(stat="identity",fill="#999999")+
  geom_vline(aes(xintercept=upper_threshold_rate,linetype="\u03B3 lower bound (0.214)"),color="#E69F00")+
  geom_hline(aes(yintercept=0.05,linetype="5% criterion"),color="red")+
  theme_classic()+
  xlab("Asocial imitation rate")+
  ggtitle("Decision Biasing bound")+
  xlim(c(-0.05,0.75))+
  ylab("Proportion")+
  scale_linetype_manual(name = "", values = c(2, 2), 
                        guide = guide_legend(override.aes = list(color = c("red","#E69F00"))))+
  theme(legend.position = c(0.7, 0.35),strip.background=element_blank(),
        legend.background=element_blank(), legend.key=element_blank()))
DB_just <- addSmallLegend(DB_just,8,7.5,1)

#VS
VS_bounding <- read.csv("./Data/VS_bounds.csv")
VS_bounding <- VS_bounding%>%group_by(alpha)%>%summarize(perc_change = mean(perc_change))
VS_criterion <- approx(VS_bounding$perc_change,VS_bounding$alpha,xout=0.05)$y

(VS_just <- ggplot(VS_bounding,aes(x=alpha,y=perc_change))+
  geom_line()+
  theme_classic()+
  ylab("Proportional effect")+ #"Effective proportion of social information"
  xlab(expression(alpha))+
  geom_hline(aes(yintercept = 0.05,linetype = "5% criterion"),color="red")+
  ggtitle("Value Shaping bound")+
  geom_vline(aes(xintercept = VS_criterion,linetype="\u03b1 lower bound (0.125)"),color="#009E73")+
  scale_linetype_manual(name = "", values = c(2, 2), 
                        guide = guide_legend(override.aes = list(color = c("red","#009E73"))))+
  theme(legend.position = c(0.65, 0.3),strip.background=element_blank(),
        legend.background=element_blank(), legend.key=element_blank()))
VS_just <- addSmallLegend(VS_just,8,7.5,1)

# SG
SG_bounding <- read.csv("./Data/SG_bounding.csv")
SG_criterion <- approx(SG_bounding$obs_prop,SG_bounding$eps_soc,xout=0.05)$y
SG_just <- ggplot(SG_bounding,aes(x=eps_soc,y=obs_prop))+
  geom_line()+
  theme_classic()+
  ylab("Proportional effect")+ #"Effective proportion of social information"
  xlab(expression(epsilon["soc"]))+
  geom_hline(aes(yintercept = 0.05, linetype = "a"),color="red")+
  geom_vline(aes(xintercept = SG_criterion,linetype="b"),color="#56B4E9")+
  ggtitle("Social Generaliazation bound")+
  scale_linetype_manual(name = "", values = c(2, 2), labels = c(a = "5% criterion",b = expression(paste(epsilon[italic(soc)]," upper bound (19)"))),
                        guide = guide_legend(override.aes = list(color = c("red","#56B4E9"))))+
  theme(legend.position = c(0.55, 0.2),legend.text.align = 0,strip.background=element_blank(),
        legend.background=element_blank(), legend.key=element_blank())
(SG_just <- addSmallLegend(SG_just,8,7.5,1))

(boundplots <- cowplot::plot_grid(DB_just,VS_just,SG_just,ncol=3,labels="auto"))
ggsave("./plots/supp_bounding.pdf",plot = boundplots, width=8.5,height=3.5)
