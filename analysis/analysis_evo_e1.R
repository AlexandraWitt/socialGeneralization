#Figs 3-5 code (+ some supps)

library(ggplot2)
library(dplyr)
library(ggtern)
library(viridis)
library(tidyr)
library(cowplot)
library(lme4)
library(lmerTest)

library(gridExtra)
library(ggsignif)
library(ggbeeswarm)
library(bayestestR)
library(viridis)
library(brms)
library(sjPlot)

saveAll <- F
cbPalette <- c("#999999","#E69F00", "#009E73","#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00")#
####extra imports and functions #######
source('statisticalTests.R')
std <- function(a) sd(a) / sqrt(length(a))

run_model <- function(expr, modelName, path=".", reuse = TRUE) {
  path <- paste0(path,'/', modelName, ".brm")
  if (reuse) {
    fit <- suppressWarnings(try(readRDS(path), silent = TRUE))
  }
  if (is(fit, "try-error")) {
    fit <- eval(expr)
    saveRDS(fit, file = path)
  }
  fit
}

formatHDI <- function(x, signDig=2){
  x.mean <- sprintf("%.*f",signDig, mean(x))
  x.CI <- sprintf("%.*f",signDig, hdi(x))
  return(paste0(x.mean, ' [', x.CI[1], ', ', x.CI[2], ']'))
}

addSmallLegend <- function(myPlot, pointSize = 0.5, textSize = 3, spaceLegend = 0.1) {
  myPlot +
    guides(shape = guide_legend(override.aes = list(size = pointSize)),
           fil = guide_legend(override.aes = list(size = pointSize))) +
    theme(legend.title = element_text(size = textSize), 
          legend.text  = element_text(size = textSize),
          legend.key.size = unit(spaceLegend, "lines"))
}


#Fig 3 + supps #########
#import
data = read.csv("./Data/evoSim.csv")


#full dataset (AS comparisons included)############

data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
counts <- data%>%group_by(gen,mix,model)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                            beta=mean(beta),tau=mean(tau),n=n(),score=mean(score))

#final evolved SG number
avg_SG <- sum(subset(counts,gen==499&model=="SG")$n)/sum(subset(counts,gen==499)$n)

#percentages split by mix
(evo_all <- ggplot(counts,aes(x=gen,y=n/1000,color=model,fill=model))+ 
    geom_line()+
    theme_classic()+
    scale_color_manual(values = cbPalette)+ 
    facet_wrap(~mix,ncol = 4)+
    ylab("Probability")+
    xlab("Generation")+
    labs(color='Model'))+
  theme(strip.background = element_blank())
if (saveAll){ggsave("./plots/evosim_full.pdf")}

#SG parameter plots for all mixes###############
means <- subset(data,model=="SG")%>%group_by(gen,mix)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                                         beta=mean(beta),tau=mean(tau),score=mean(score))

means$n <- subset(counts,model=="SG")$n
means$mix = factor(means$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                      "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
avg_lam <- mean(subset(means,gen==499)$lambda)
avg_bet <- mean(subset(means,gen==499)$beta)
avg_tau <- mean(subset(means,gen==499)$tau)
avg_eps <- mean(subset(means,gen==499)$eps_soc)

# just social model comparisons: Fig 3 ##########

cbPalette_noAS <- c("#E69F00", "#009E73","#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00")#"#999999",
data <- filter(data,mix %in% c("DB","VS","SG","DB.VS","DB.SG","VS.SG","DB.VS.SG"))
counts <- data%>%group_by(gen,mix,model)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                            beta=mean(beta),tau=mean(tau),n=n(),score=mean(score))
wide_counts <- pivot_wider(counts,id_cols=c(gen,mix),names_from=model,values_from = n)
wide_counts <- wide_counts <- replace_na(wide_counts,list(DB=0,VS=0,SG=0))


(tern <- ggtern(wide_counts,aes(DB,VS,SG,color=mix))+
    geom_path(aes(group = mix), alpha = 0.8,lwd=1.5)+
    geom_label( data = subset(wide_counts, gen == min(gen)), aes(label = mix), alpha = 0.8, show.legend=FALSE)+
    geom_point( data = subset(wide_counts, gen == max(gen)), shape = 4, size  =3,stroke=1.5)+
    theme_classic()+
    theme_showarrows()+
    scale_colour_manual(values=cbPalette_noAS)+
    theme_nomask()+
    theme(legend.position = 'none'))
if (saveAll){ggsave("./plots/tern.pdf")}

#SG parameter plots for those mixes ###################
means <- subset(data,model=="SG")%>%group_by(gen,mix)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                                         beta=mean(beta),tau=mean(tau))

means$n <- subset(counts,model=="SG")$n
means$mix = factor(means$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                      "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))

#generalization
(p1 <- ggplot(means,aes(x=gen,y=lambda,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1)+
    theme_classic()+
    scale_color_manual(values = cbPalette_noAS) +
    scale_fill_manual(values = cbPalette_noAS) +
    #ylab(expression(lambda))+
    ylab("Generalization")+
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction="horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')+
    ylim(0,3))
if (saveAll){ggsave("./plots/evosim_lambda.pdf")}

#directed exploration
(p2 <- ggplot(means,aes(x=gen,y=beta,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1)+
    theme_classic()+
    scale_color_manual(values = cbPalette_noAS) +
    scale_fill_manual(values = cbPalette_noAS) +
    ylab("\u03b2")+
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction = "horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')) 
if (saveAll){ggsave("./plots/evosim_beta.pdf")}

#random exploration
(p3 <- ggplot(means,aes(x=gen,y=tau,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1)+
    theme_classic()+
    scale_color_manual(values=cbPalette_noAS) +
    scale_fill_manual(values=cbPalette_noAS) +
    #ylab(expression(tau))+
    ylab("Random exploration")+
    theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')) 
if (saveAll){ggsave("./plots/evosim_tau.pdf")}

#social noise
(p4 <- ggplot(means,aes(x=gen,y=eps_soc,color=mix))+
    geom_line(alpha=0.6)+
    stat_summary(fun=mean,geom="line",color="black",lwd=1)+
    theme_classic()+
    scale_color_manual(values = cbPalette_noAS) +
    scale_fill_manual(values = cbPalette_noAS) +
    #ylab(expression(epsilon["soc"]))+
    ylab("Social noise")+
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction = "horizontal",legend.position = "None")+ #
    xlab("Generation")+
    labs(color='Initial \npopulation'))
if (saveAll){ggsave("./plots/evosim_eps.pdf")}

#combine
cowplot::plot_grid(p1,p2,p3,p4)
if (saveAll){ggsave("./plots/evolved_pars.png")}

#Percentage SG over generations
ggplot(subset(counts,model=="SG"),aes(x=gen,y=n/1000,color=mix))+
  geom_line()+
  stat_summary(fun=mean,geom="line",color="black",lwd=1.25)+
  theme_classic()+
  xlab("Generation")+
  theme(legend.position = "None",axis.title.x = element_text(size = 10),axis.title.y = element_text(size = 10))+#c(0.75,0.5))+ #legend.direction="horizontal",
  scale_color_manual(values = cbPalette_noAS) +
  ylab("p(SG)")
if (saveAll){ggsave("./plots/SG_over_time.pdf",width=2.125,height=1.5)}#,dpi=300

#Fig 4 - behaviour #############
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9",  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
saveAll <- F

#demography #########
demo <- read.csv("./Data/e1_demo.csv")
meanAge <- mean(as.numeric(demo$Age),na.rm=T)
sdAge <- sd(as.numeric(demo$Age),na.rm=T)

meanCompletion <- mean(demo$Time.taken)/60
semCompletion <- std(demo$Time.taken)/60

meanPayout <- mean(demo$totalPayment)
semPayout <- std(demo$totalPayment)

table(demo$Sex)

#data imports + sanity check LCs + overall coherence##########
pilot_data <- read.csv("./Data/e1_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]

pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()
randomChoicePerc <- mean(subset(pilot_data,trial!=0)$isRandom)

meandata <- pilot_data%>%group_by(group,trial)%>%summarise(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                           search_dist=mean(search_dist),variance=mean(variance),
                                                           soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                           soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))


(lc <- ggplot(meandata,aes(x=trial,y=meanReward))+
    geom_line(aes(color=factor(group)),alpha=0.25)+
    geom_hline(yintercept=0.5, linetype="dashed",color="red",lwd=1)+
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = 0.2, color = NA)+
    stat_summary(fun=mean,geom="line",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(length(unique(meandata$group)))) +
    theme(legend.position="None")+
    ylab("Average Reward")+
    xlab("Trial"))
if (saveAll){ggsave("./plots/lc.pdf")}

meandata <- pilot_data%>%group_by(group,trial)%>%summarise(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                           search_dist=mean(search_dist),variance=mean(variance),
                                                           soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                           soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))


(soc_sd_trial <- ggplot(meandata,aes(x=trial,y=soc_sd))+
    geom_line(aes(color=factor(group)),alpha=0.25)+
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = 0.2, color = NA)+
    stat_summary(fun=mean,geom="line",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(length(unique(meandata$group)))) +
    theme(legend.position="None")+
    ylab("Social search distance")+
    xlab("Trial"))
if (saveAll){ggsave("./plots/soc_sd_pilot.pdf")}
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)


meandata <- pilot_data%>%group_by(id)%>%summarise(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                           search_dist=mean(search_dist),variance=mean(variance),
                                                           soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                           soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))
#better than chance performance
ttestPretty(meandata$meanReward,mu=0.5)
#closer than random distance
randDist = mean(sqrt((sample(1:11, 10000, replace=TRUE) - sample(1:11, 10000, replace=TRUE))^2 + (sample(1:11, 10000, replace=TRUE) - sample(1:11, 10000, replace=TRUE))^2))
ttestPretty(meandata$soc_sd,mu=randDist)

# Social search distance ~ previous reward ##############
#Human Data
data = read.csv("./Data/e1_data_regressable.csv")
data$social = factor(data$social)
data <- subset(data,isRandom==0)

#regression
dist_prev_rew = run_model(brm(soc_sd ~ soc_rew * social + (1+soc_rew+social|group/agent),
                              data = data, cores = 4, iter = 4000, warmup = 1000,
                              control = list(adapt_delta = 0.99,max_treedepth = 20)), modelName = 'dist_prev_rew_e1')

#posterior samples
post <- dist_prev_rew %>% brms::posterior_samples()
formatHDI(post$b_soc_rew) #individual info
formatHDI(post$b_soc_rew+post$`b_soc_rew:social1`+post$b_social1) #social info

#generate prediction set
prev_rew = seq(0,50)/50
test <- expand.grid(soc_rew = prev_rew,social=levels(data$social))
preds = fitted(dist_prev_rew, re_formula=NA,newdata=test,probs=c(.025,.975))
plotdata = data.frame(prev_rew=test$soc_rew,social=test$social,sdist=preds[,1],se=preds[,2],lower=preds[,3],upper=preds[,4])

socAsoc <- c("#000000","#1F78B4")

(p <- ggplot()+
    stat_summary(data,mapping=aes(x = round(soc_rew*33)/33,y=soc_sd,color=factor(social),fill=factor(social)),fun = mean,geom='point',alpha=0.8)+
    geom_line(plotdata,mapping=aes(x=prev_rew,y=sdist,color=factor(social)),lwd=1.25)+
    geom_ribbon(plotdata,mapping=aes(x=prev_rew,y=sdist,ymin=lower,ymax=upper,fill=factor(social)),alpha=0.3)+
    scale_color_manual(values=socAsoc,name="Information source",labels=c("Individual","Social","Simulated asocial\nbaseline"))+
    scale_fill_manual(values=socAsoc,name="Information source",labels=c("Individual","Social","Simulated asocial\nbaseline"))+
    theme_classic()+
    xlab("Previous Reward")+
    ylab("Search distance")+
    theme(legend.position = c(0.75,0.8), legend.background = element_blank(), legend.key = element_blank()))
if (saveAll){ggsave("./plots/soc_sd_prev_rew_e1.pdf",plot=p)}


#Social search distance split####
data$dist <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"soc_sd"]==0,"0",ifelse(data[x,"soc_sd"]==1,"1",ifelse(data[x,"soc_sd"]<3,"2",">=3"))))
data$dist <- factor(data$dist,levels=c("0","1","2",">=3"))

data$socvalue <- data$soc_rew - data$prev_rew
data$rew_bins <- round(data$socvalue*100)/100
pdata <- subset(data,social==1&!is.na(soc_rew)) #social==1&dist=='0'|social==1&dist=='1'
pdata$socvalue <- pdata$soc_rew - pdata$prev_rew
freqs <- subset(pdata)%>%group_by(rew_bins,dist)%>%summarize(n=n(),dist=unique(dist))%>%mutate(freq=n/sum(n))
freqs <- left_join(freqs, pdata %>% count(dist) %>% mutate(base_freq = n / sum(n)) %>% select(dist, base_freq), by = "dist")
freqs <- left_join(freqs,pdata %>% count(rew_bins) %>% mutate(rew_bin_freq = n / sum(n)) %>% select(rew_bins, rew_bin_freq), by = "rew_bins")
freqs$freq <- freqs$freq/freqs$rew_bin_freq
freqs$dist <- factor(freqs$dist,levels=c("0","1","2",">=3"))
colors <- c("#ff7f00","#425e8a")
(soc_sd <- ggplot(subset(freqs,dist=="0"|dist=="1"),aes(x=rew_bins,y=freq/base_freq,color=dist,fill=dist))+ # subset(freqs,dist!=">=3")
  geom_point(alpha=0.8)+
  geom_smooth(method="lm")+
  theme_classic()+
  ylab("P(soc. search dist.)")+
  xlab("Previous social reward")+
  xlim(c(0.5,1))+
  theme(legend.position = c(0.35,0.75),legend.background = element_blank(),
        legend.key = element_blank())+
  scale_fill_manual(values = colors,name="Search distance")+
  scale_color_manual(values = colors,name="Search distance")) #,end=.8

soc_sd_test <- lm(freq/base_freq~rew_bins*dist,data=subset(freqs,dist=="0"&rew_bins>0.5|dist=="1"&rew_bins>0.5))
summary(soc_sd_test)

soc_sd_split = run_model(brm(freq~rew_bins*dist,
                              data = subset(freqs,dist=="0"|dist=="1"), cores = 4, iter = 4000, warmup = 1000,
                              control = list(adapt_delta = 0.99,max_treedepth = 20)), modelName = 'soc_sd_split_e1')

post <- soc_sd_split %>% brms::posterior_samples()
formatHDI(post$b_rew_bins) #reward on imitation
formatHDI(post$`b_rew_bins:dist1`) #reward on innovation

#Fig. 5 (modelling results) --------------------------------

################################
#Fitting figure
################################
#protected exceedance probability
data = read.csv("./Data/pxp_e1.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9", "#CC79A7",  "#F0E442", "#0072B2", "#D55E00")

(pxp <- ggplot(data,aes(x=model,y=exceedance,color=model,fill=model))+
    geom_bar(stat="identity")+
    geom_hline(yintercept=0.25,linetype="dashed",color="red",lwd=1)+
    theme_classic()+
    scale_colour_manual(values=cbPalette)+
    scale_fill_manual(values=cbPalette)+
    xlab("Model")+
    ylab("p(best model) (pxp)")+
    theme(legend.position="None"))

#fit parameters
data <- read.csv("./Data/fit+pars_e1_nLL.csv")
data$model <- factor(data$model,levels=c("AS","DB","VS","SG"))

pilot_data <- read.csv("./Data/e1_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]

pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(agent,group)%>%summarise(meanReward =mean(reward),soc_sd=mean(soc_sd,na.rm=T))
data <- merge(meandata,data,by=c("agent","group"))
#only evaluate parameters for participants best fit by SG
data$SG_best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$SG_best <- factor(data$SG_best)

#parameters
#generalization
(lam <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=lambda,color="#56B4E9"))+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    scale_colour_manual(values="#56B4E9")+
    theme_classic()+
    xlab("Generalization")+
    #ylab(expression(lambda))+
    ylab("Parameter value")+
    theme(legend.position = "None",axis.text.x = element_blank(),axis.ticks.x = element_blank()))

avg_lamb <- mean(subset(data,model=="SG"&SG_best==1)$lambda)
ttestPretty(subset(data,model=="SG"&SG_best==1)$lambda,mu=2)

#directed exploration
(bet <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=beta,color="#56B4E9"))+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    scale_colour_manual(values="#56B4E9")+
    theme_classic()+
    xlab("Directed exploration")+
    #ylab(expression(beta))+
    #ylab("Directed exploration")+
    ylab("Parameter value")+
    theme(legend.position = "None",axis.text.x = element_blank(),axis.ticks.x = element_blank()))

avg_bet <- mean(subset(data,model=="SG"&SG_best==1)$beta)
ttestPretty(subset(data,model=="SG"&SG_best==1)$beta,mu=0.5)

#random exploration
(tau <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=tau,color="#56B4E9"))+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    scale_colour_manual(values="#56B4E9")+
    theme_classic()+
    xlab("Random exploration")+
    #ylab(expression(tau))+
    #ylab("Random exploration")+
    ylab("Parameter value")+
    theme(legend.position = "None",axis.text.x = element_blank(),axis.ticks.x = element_blank()))

avg_tau <- mean(subset(data,model=="SG"&SG_best==1)$tau)

#social noise
(eps_soc <- ggplot(subset(data,data$model=="SG"&SG_best==1),aes(x=model,y=par,color="#56B4E9"))+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    scale_colour_manual(values="#56B4E9")+
    theme_classic()+
    xlab("Social noise")+
    #ylab(expression(epsilon["soc"]))+
    #ylab("Social noise")+
    ylab("Parameter value")+
    theme(legend.position = "None",axis.text.x = element_blank(),axis.ticks.x = element_blank()))

avg_eps <- mean(subset(data,model=="SG"&SG_best==1)$par)
ttestPretty(subset(data,model=="SG"&SG_best==1)$par,mu=3.29)

(pars <- cowplot::plot_grid(lam,bet,tau,eps_soc,nrow=2))
pars <- ggdraw(add_sub(pars, "SG Parameters", vpadding=grid::unit(0,"lines"),y=6, x=0.55, vjust=4.5,size=12))

#reward over eps soc
eps_rew_test <- cor.test(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$meanReward,method = "kendall")
label = c(paste0("r[tau]==",round(eps_rew_test$estimate,2)),paste0("p==",round(eps_rew_test$p.value,4)))
(eps_soc_rew_e1 <- ggplot(subset(data,data$model=="SG"&data$SG_best==1),aes(x=par,y=meanReward))+
    geom_point()+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    theme_classic()+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    ylab("Mean Reward")+
    annotate("text",x=15,y=c(0.9,0.88),label = label,parse=T)+
    theme(legend.position = c(0.625,0.83)))

corTestPretty(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$meanReward,method = "kendall")

#beta over eps_soc
beta_expl_test <- cor.test(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$beta,method = "kendall")
label = c(paste0("r[tau]==",round(beta_expl_test$estimate,2)),paste0("p==",round(beta_expl_test$p.value,3)))

(expl_replace <- ggplot(subset(data, model=="SG"&SG_best==1),aes(x=par,y=beta))+
  geom_point()+
  theme_classic()+
  geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
  #xlab(expression(epsilon["soc"]))+
  xlab("Social noise")+
  #ylab(expression(beta))+
  ylab("Directed exploration")+
  annotate("text",x=5,y=c(0.55,0.51),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$beta,method = "kendall")

(all_res <- cowplot::plot_grid(cowplot::plot_grid(lc,soc_sd_trial,p,soc_sd,nrow=1,labels="auto"),
                              cowplot::plot_grid(pxp,pars,expl_replace,eps_soc_rew_e1,nrow=1,labels=c("e","f","g","h"),rel_widths = c(0.5,1.5,1,1)),nrow=2)
)
ggsave("./plots/all_res_e1.pdf",plot=all_res,width=14,height=7)
