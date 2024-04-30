#Supplementary figures
#Alex 2023

library(ggplot2)
library(dplyr)
library(ggtern)
library(viridis)
library(tidyr)
library(cowplot)
library(lme4)
library(lmerTest)
library(ggbeeswarm)

library(ggeffects)
library(ggsignif)
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

#Prior plots ===============

x <- seq(0,10,0.0001)
lambda <- rlnorm(1000000,-0.75,0.5)
beta <- rlnorm(1000000,-0.75,0.5)
tau <- rlnorm(1000000,-4.5,0.9)
alpha <- runif(1000000)
eps_soc <- rexp(1000000,0.5)
gamma <- runif(1000000)


priors <- data.frame(lambda,beta,tau,alpha,eps_soc,gamma)

#generalization
p1 <- ggplot(priors,aes(x=lambda))+
  geom_density()+
  geom_vline(aes(xintercept=mean(lambda),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3,limits=c(0,2.1))+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  xlab("Generalization")+
  #xlab(expression(lambda))+
  ylab("p")

#directed exploration
p2 <-ggplot(priors,aes(x=beta))+
  geom_density()+
  geom_vline(aes(xintercept=mean(beta),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3,limits=c(0,2.1))+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  #xlab(expression(beta))+
  xlab("Dir. exploration")+
  ylab("")

#random exploration
p3 <-ggplot(priors,aes(x=tau))+
  geom_density()+
  geom_vline(aes(xintercept=mean(tau),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3,limits=c(0,0.1))+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  #xlab(expression(tau))+
  xlab("Rand. exploration")+
  ylab("")

#social bonus (VS)
p4 <-ggplot(priors,aes(x=alpha))+
  geom_density(color="#009E73")+
  geom_vline(aes(xintercept=mean(alpha),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3)+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  #xlab(expression(alpha))+
  ylab("")+
  xlab("Social bonus")

#social noise (SG)
p5 <- ggplot(priors,aes(x=eps_soc))+
  geom_density(color="#56B4E9")+
  geom_vline(aes(xintercept=mean(eps_soc),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3,limits = c(0,15))+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  #xlab(expression(epsilon["soc"]))+
  ylab("")+
  xlab("Social noise")

#imitation bias (DB)
p6 <-ggplot(priors,aes(x=gamma))+
  geom_density(color="#E69F00")+
  geom_vline(aes(xintercept=mean(gamma),color="red"))+
  theme_classic()+
  scale_x_continuous(n.breaks = 3)+
  theme(legend.position = "None",plot.title = element_text(size=12,hjust=0),
        axis.text.y = element_blank(),axis.ticks = element_blank())+
  #xlab(expression(gamma))+
  ylab("")+
  xlab("Imitation rate")

#combine
cowplot::plot_grid(p1, p2, p3,p6,p4,p5,nrow = 1)
if (saveAll){ggsave("./plots/priors_wide.pdf",width=8.75,height=1.25)}


#bonus evoSim
#full dataset (AS comparisons included) ###########
data = read.csv("./Data/evoSim.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
counts <- data%>%group_by(gen,mix,model)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                            beta=mean(beta),tau=mean(tau),n=n(),score=mean(score))

#final evolved SG number
avg_SG <- sum(subset(counts,gen==499&model=="SG")$n)/sum(subset(counts,gen==499)$n)

#percentages split by mix
(evo_all_canon <- ggplot(counts,aes(x=gen,y=n/1000,color=model,fill=model))+ 
    geom_line()+
    theme_classic()+
    scale_color_manual(values = cbPalette)+ 
    facet_wrap(~mix,ncol = 4)+
    ylab("Probability")+
    xlab("Generation")+
    labs(color='Model'))+
  theme(strip.background = element_blank())
if (saveAll){ggsave("./plots/evosim_full.pdf")}

#SG parameter plots for all mixes ###############
means <- subset(data,model=="SG")%>%group_by(gen,mix)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                                         beta=mean(beta),tau=mean(tau))

means$n <- subset(counts,model=="SG")$n
means$mix = factor(means$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                      "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))

#generalization
(p1 <- ggplot(means,aes(x=gen,y=lambda,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(15)) +
    scale_fill_manual(values = rainbow(15)) +
    ylab(expression(lambda))+
    #ylab("Generalization")+ 
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction="horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')+
    ylim(0,3))

#directed exploration
(p2 <- ggplot(means,aes(x=gen,y=beta,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(15)) +
    scale_fill_manual(values = rainbow(15)) +
    ylab(expression(beta))+
    #ylab("Directed\nexploration")+ #"\u03b2"
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction = "horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')) 

#random exploration
(p3 <- ggplot(means,aes(x=gen,y=tau,color=mix))+
    geom_line()+
    stat_summary(fun=mean,geom="line",color="black",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(15)) +
    scale_fill_manual(values = rainbow(15)) +
    ylab(expression(tau))+
    #ylab("Random\nexploration")+ 
    theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
    xlab("Generation")+
    labs(color='Initial \npopulation')) 

#social noise
(p4 <- ggplot(means,aes(x=gen,y=eps_soc,color=mix))+
    geom_line(alpha=0.6)+
    stat_summary(fun=mean,geom="line",color="black",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(15)) +
    scale_fill_manual(values = rainbow(15)) +
    ylab(expression(epsilon["soc"]))+
    #ylab("Social noise")+ 
    theme(axis.title.y = element_text(angle = 0,hjust=0.5,vjust=0.5),legend.direction = "horizontal",legend.position = "None")+ #
    xlab("Generation")+
    labs(color='Initial \npopulation'))

#combine
evo_pars <- cowplot::plot_grid(p1,p2,p3,p4,nrow=4)

combo <- cowplot::plot_grid(evo_all_canon, evo_pars, nrow=1, labels = "auto",rel_widths = c(0.75,0.25))
ggsave("./plots/evoSim_details_stack.pdf", plot = combo, width = 10, height = 5.5)

# Najar replication =====================
data = read.csv("./Data/evoSim_Najar.csv")

data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
counts <- data%>%group_by(gen,mix,model)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                            beta=mean(beta),tau=mean(tau),n=n(),score=mean(score))

#final evolved VS&SG 
avg_SG_Najar <- sum(subset(counts,gen==499&model=="SG")$n)/sum(subset(counts,gen==499)$n)
avg_VS_Najar <- sum(subset(counts,gen==499&model=="VS")$n)/sum(subset(counts,gen==499)$n)

#percentages split by mix
(evo_all_naj <- ggplot(counts,aes(x=gen,y=n/1000,color=model,fill=model))+ 
    geom_line()+
    theme_classic()+
    scale_color_manual(values = cbPalette)+ 
    facet_wrap(~mix,ncol = 4)+
    ylab("Probability")+
    xlab("Generation")+
    labs(color='Model'))+
  theme(strip.background = element_blank())
if (saveAll){ggsave("./plots/evosim_Najar.pdf")}

# social comps only #################
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
if (saveAll){ggsave("./plots/tern_Najar.pdf")}

# Najar relevant models only ##################
cbPalette_Najar <- c("#999999","#E69F00", "#009E73", "#CC79A7",  "#F0E442", "#0072B2", "#D55E00")

data = read.csv("./Data/evoSim_Najar.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))

data <- filter(data,mix %in% c("AS","DB","VS","AS.DB","AS.VS","DB.VS","AS.DB.VS"))
data$mix <- factor(data$mix,levels=c("AS","DB","VS","AS.DB","AS.VS","DB.VS","AS.DB.VS"))
counts <- data%>%group_by(gen,mix,model)%>%dplyr::summarize(alpha=mean(alpha),eps_soc=mean(eps_soc),gamma=mean(gamma),lambda = mean(lambda),
                                                            beta=mean(beta),tau=mean(tau),n=n(),score=mean(score))
wide_counts <- pivot_wider(counts,id_cols=c(gen,mix),names_from=model,values_from = n)
wide_counts <- wide_counts <- replace_na(wide_counts,list(DB=0,VS=0,AS=0))

(tern_naj <- ggtern(wide_counts,aes(AS,DB,VS,color=mix))+
    geom_path(aes(group = mix), alpha = 0.8,lwd=1.5)+
    geom_label( data = subset(wide_counts, gen == min(gen)), aes(label = mix), alpha = 0.8, show.legend=FALSE)+
    geom_point( data = subset(wide_counts, gen == max(gen)), shape = 4, size  =3,stroke=1.5)+
    theme_classic()+
    theme_showarrows()+
    scale_colour_manual(values=cbPalette_Najar)+
    theme_nomask()+
    theme(legend.position = 'none'))
if (saveAll){ggsave("./plots/tern_Najar_OG.pdf")}


#full S2
tern_naj <- ggplot_gtable(ggplot_build(tern_naj))
(S2 <- cowplot::plot_grid(evo_all_canon,evo_pars,
                          print(tern_naj),print(tern),evo_all,labels="auto",ncol=2))
ggsave("./plots/S2_attempt.pdf",width=15,height=20)


######################order and time effects######################
#exp1 round effects
pilot_data <- read.csv("./Data/e1_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]
pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()

meandata <- pilot_data%>%group_by(round,group)%>%dplyr::summarize(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                           search_dist=mean(search_dist),variance=mean(variance),
                                                           soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                           soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))

(lc_round_e1 <- ggplot(meandata,aes(x=round,y=meanReward))+
    geom_hline(yintercept=0.5,color="red",linetype="dashed")+
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = 0.2, color = NA)+
    stat_summary(fun=mean,geom="line",lwd=1.25)+
    theme_classic()+
    scale_color_manual(values = rainbow(length(unique(meandata$group)))) +
    theme(legend.position="None")+
    ylab("Average Reward")+
    xlab("Round")+
    ggtitle("Exp.1")+
    ylim(c(0.5,1)))
if (saveAll){ggsave("./plots/lc_rounds.pdf")}

#exclude random choices from analysis
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)
meandata <- pilot_data%>%group_by(round,id)%>%dplyr::summarize(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                        search_dist=mean(search_dist),variance=mean(variance),
                                                        soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                        soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))


round_effect <- lmer(meanReward~round+(1|id),data=meandata)
summary(round_effect)

#exp2 round and order effects
pilot_data <- read.csv("./Data/e2_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]
pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()

pilot_data$blockOrder <- factor(pilot_data$blockOrder,levels=c("S-I","I-S"))
meandata <- pilot_data%>%group_by(group,round,blockOrder)%>%dplyr::summarize(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                                      search_dist=mean(search_dist),variance=mean(variance),
                                                                      soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                                      soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))
meandata$blockOrder <- recode_factor(meandata$blockOrder, "I-S" = "Solo first", 
                                     "S-I" = "Group first")
meandata$blockOrder <- factor(meandata$blockOrder,levels=c("Group first","Solo first"))

(lc_round_e2 <- ggplot(meandata,aes(x=round,y=meanReward,linetype=blockOrder))+
    geom_hline(yintercept=0.5,color="red",linetype="dashed")+
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', mapping=aes(fill=blockOrder),alpha = 0.2, color = NA)+
    stat_summary(fun=mean,geom="line",lwd=1.25,mapping=aes(color=blockOrder))+
    theme_classic()+
    scale_color_manual(values = c("black","red"),name ="Block order") +
    scale_fill_manual(values= c("black","red"), name = "Block order")+
    scale_linetype_manual(values=c("solid","dotted"),name="Block order")+
    theme(legend.position=c(0.8,0.25))+
    ylab("Average Reward")+
    ylim(c(0.5,1))+
    xlab("Round")+
    ggtitle("Exp.2")+
    theme(strip.background=element_blank(),
          legend.background=element_blank(), legend.key=element_blank()))

#exclude random choices for analysis
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)
meandata <- pilot_data%>%group_by(id,round,blockOrder)%>%dplyr::summarize(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                                   search_dist=mean(search_dist),variance=mean(variance),
                                                                   soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                                   soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))


blockOrderEff <- lmer(meanReward~round*blockOrder+(1|id),data=meandata)#data=subset(meandata,round!=0))
summary(blockOrderEff)

ordereffs <- cowplot::plot_grid(lc_round_e1,lc_round_e2,nrow=1,labels="auto")
ggsave("./plots/supp_ordereffects.pdf",width=6,height=3,plot=ordereffs)

#Improvement potential #####################################
#Exp.1
data = read.csv("./Data/e1_data_regressable.csv")
data$social = factor(data$social)
data <- subset(data,isRandom==0)

#value of social information is in its difference from previous individual reward, individual improvement potential is 1-prev_rew
data[data$social==1,"soc_value"] <- data[data$social==1,"soc_rew"]-data[data$social==1,"prev_rew"]
data[data$social==0,"soc_value"] <- 1-data[data$social==0,"prev_rew"]

expl_eff = run_model(brm(expl_eff ~ soc_value * social + trial + (1 + soc_value + social + trial | agent/group),
                         data = subset(data,soc_value>=0.1), cores = 4, iter = 4000, warmup = 1000,
                         control = list(adapt_delta = 0.99,max_treedepth = 20)), modelName = 'e1_expl_eff')


(reg_pars_ee1 <- plot_model(expl_eff, axis.labels =c('Improvement potential:\nSocial info', 'Trial', 'Social\ninfo', 'Improvement potential'), bpe = "mean", bpe.style = "dot", bpe.color='black', show.values = TRUE, vline.color='grey',  ci.lvl=.95, sort.est=FALSE, show.p=FALSE) +
    theme_classic()+
    xlab('')+
    ggtitle("Reward improvement"))

post <- expl_eff %>% brms::posterior_samples() #posterior samples
formatHDI(post$b_soc_value) #group effect previous reward
formatHDI(post$b_social1)
formatHDI(post$`b_soc_value:social1`)

#predict model
preds = ggpredict(expl_eff,c("soc_value","social"))
plotdata = as.data.frame(preds)

socAsoc <- c("#999999", "#0072B2")

(imp1 <- ggplot()+
    stat_summary(data,mapping=aes(x=round(soc_value*50)/50,y=expl_eff,color=factor(social),fill=factor(social)),fun = mean,geom='point',alpha=0.8)+
    geom_line(plotdata,mapping=aes(x=x,y=predicted,color=factor(group)),lwd=1.25)+
    geom_ribbon(plotdata,mapping=aes(x=x,y=predicted,ymin=conf.low,ymax=conf.high,fill=factor(group)),alpha=0.3)+
    scale_color_manual(values=socAsoc,name="Information source",labels=c("Individual","Social"))+
    scale_fill_manual(values=socAsoc,name="Information source",labels=c("Individual","Social"))+
    theme_classic()+
    xlab("Improvement potential")+
    ylab("Reward improvement")+
    theme(legend.position = c(0.35,0.65), legend.background = element_blank(), legend.key = element_blank()))

#Improvement potential exp2#####################################
data = read.csv("./Data/e2_data_regressable.csv")
data$social = factor(data$social)
data$taskType <- factor(data$taskType)
data <- subset(data,isRandom==0)

data[data$social==1,"soc_value"] <- data[data$social==1,"soc_rew"]-data[data$social==1,"prev_rew"]
data[data$social==0,"soc_value"] <- 1-data[data$social==0,"prev_rew"]

expl_eff = run_model(brm(expl_eff ~ soc_value*social*taskType +trial+(1|group/agent),
                         data = subset(data,soc_value>=0.1), cores = 4, iter = 4000, warmup = 1000,
                         control = list(adapt_delta = 0.99,max_treedepth = 20)), modelName = 'e2_expl_eff')


(reg_pars_ee2 <- plot_model(expl_eff, bpe = "mean", bpe.style = "dot", bpe.color='black',
                            axis.labels =c('Improvement potential.:\nSocial info:\nGroup round', 'Social info:\nGroup round', 'Improvement potential:\nGroup round', 'Improvement potential:\nSocial info','Trial', 'Group round' ,'Social info', 'Improvement potential'),
                            show.values = TRUE, vline.color='grey',  ci.lvl=.95, sort.est=FALSE, show.p=FALSE) +
    theme_classic()+
    xlab('')+
    ggtitle("Reward improvement"))

post <- expl_eff %>% brms::posterior_samples() #posterior samples
formatHDI(post$b_soc_value) #group effect previous reward
formatHDI(post$b_social1)
formatHDI(post$`b_soc_value:social1`)

#
formatHDI(post$`b_soc_value:taskTypesocial`) #imp potx group round
formatHDI(post$`b_social1:taskTypesocial`) #social info x group round
formatHDI(post$`b_soc_value:social1:taskTypesocial`) #imp pot x social info x group round 

#predict model
preds = ggpredict(expl_eff,c("soc_value","social","taskType"))
plotdata = as.data.frame(preds)

exp2pal <- c("#999999","#A6CEE3","#000000","#1F78B4")

soc_value <- seq(0,50)/50
test <- expand.grid(soc_value = soc_value,social=levels(data$social),taskType=levels(data$taskType),trial=min(data$trial):max(data$trial))
preds = fitted(expl_eff, re_formula=NA,resp=c("soc_value","social","taskType"),newdata=test,probs=c(.025,.975))
plotdata = data.frame(soc_value=test$soc_value,social=test$social,taskType=test$taskType,expl_eff=preds[,1],se=preds[,2],lower=preds[,3],upper=preds[,4])
plotdata <- plotdata%>%group_by(social,taskType,soc_value)%>%dplyr::summarize(expl_eff=mean(expl_eff),se=mean(se),lower=mean(lower),upper=mean(upper))
plotdata$int <- interaction(plotdata$taskType,factor(plotdata$social))
plotdata$int <- factor(plotdata$int,levels = c("individual.0","individual.1","social.0","social.1"))
plotdata$int <- recode_factor(plotdata$int, individual.0 = "Solo + individual", 
                              individual.1 = "Solo + social", social.0= "Group + individual",
                              social.1 = "Group + social")

data$int <- interaction(data$taskType,factor(data$social))
data$int <- factor(data$int,levels = c("individual.0","individual.1","social.0","social.1"))
data$int <- recode_factor(data$int, individual.0 = "Solo + individual", 
                          individual.1 = "Solo + social", social.0= "Group + individual",
                          social.1 = "Group + social")


(imp2 <- ggplot()+
    stat_summary(data,mapping=aes(x=round(soc_value*20)/20,y=expl_eff,color=int,fill=int),fun = mean,geom='point',alpha=0.8)+
    geom_line(plotdata,mapping=aes(x=soc_value,y=expl_eff,color=int),lwd=1.25)+
    geom_ribbon(plotdata,mapping=aes(x=soc_value,y=expl_eff,ymin=lower,ymax=upper,fill=int),alpha=0.3)+
    scale_color_manual(values=exp2pal,name="Task type + info source")+
    scale_fill_manual(values=exp2pal,name="Task type + info source")+   
    theme_classic()+
    xlab("Improvement potential")+
    ylab("Reward improvement")+
    theme(legend.position= c(0.3,0.7),legend.background = element_blank(), legend.key = element_blank()))

imp_supp <- cowplot::plot_grid(imp1, reg_pars_ee1,imp2,reg_pars_ee2,labels="auto",nrow=2)
ggsave("./plots/imp_supp.pdf",width=14,height=7)


####################best fit######

# data <- read.csv("./Data/fit+pars_poisson_nLL_mean.csv")
# data$model <- factor(data$model,levels = c("AS","DB","VS","SG"))
# data[data$r2<0,"r2"]=0
# 
# ggplot(data, aes(x=model,y=r2,color=model))+
#   geom_beeswarm(alpha=0.4)+
#   geom_boxplot()+
#   # stat_summary(fun.data = mean_cl_normal,
#   #              geom = "linerange",color="black") +
#   # stat_summary(fun=mean, color="black",geom="point",
#   #              shape=16, size=0.75,show_guide = FALSE)+
#   theme_classic()+
#   scale_color_manual(values=cbPalette)+
#   scale_fill_manual(values=cbPalette)

data <- read.csv("./Data/fit+pars_e1_nLL.csv")
model_perf_e1 <- data%>%group_by(model)%>%dplyr::summarize(pseudo_r2 = mean(r2))
data <- read.csv("./Data/fit+pars_e1.csv")
nLL_e1 <- data%>%group_by(model)%>%dplyr::summarize(nLL_mean = mean(fit))
model_perf_e1$nLL_mean <- nLL_e1$nLL_mean

data <- read.csv("./Data/fit+pars_e1_nLL.csv")
data <- subset(data,model=="AS")
data$fit_AS <- data$fit_AS/data$fit_SG
data$fit_DB <- data$fit_DB/data$fit_SG
data$fit_VS <- data$fit_VS/data$fit_SG
data$fit_SG <- data$fit_SG/data$fit_SG
data <- subset(data, select=-c(model))

data <- data%>%pivot_longer(cols=c(starts_with("fit_")),
                    names_to = "model",
                    values_to = "fit",
                    names_prefix = "fit_")
data$model <- factor(data$model,levels = c("AS","DB","VS","SG"))
data <- distinct(data)


(e1 <- ggplot(subset(data,model!="SG"),aes(x=model,y=fit,color=model,fill=model))+
  #geom_quasirandom(alpha = 0.8,width = 0.2)+
  #geom_line(aes(group=id),alpha=0.5)+
  stat_summary(aes(group=1),geom='point',fun=mean)+
  stat_summary(fun.data = mean_se, geom = "errorbar",width=0.3)+
    #geom_bar(stat="identity")+
  #  geom_boxplot(fill=NA)+
  geom_hline(yintercept=1, color="red",linetype="dashed")+
  scale_color_manual(values=cbPalette,name="Model")+
  scale_fill_manual(values=cbPalette,name="Model")+
  ylab("nLL ratio of model/SG")+
  #coord_cartesian(ylim=c(1, 1.1))+
  #ylim(c(0.75,1.25))+
  xlab("Model")+
  ggtitle("Exp. 1")+
  theme_classic()+
  theme(legend.position="none"))
ggsave("./plots/e1_nLL_diff_full.pdf",width=5,height=3.5)

ggplot(subset(data,model!="SG"),aes(x=model,y=fit,color=model))+
  geom_point()+
  geom_line(aes(group=id),alpha=0.5)+
  stat_summary(aes(group=1),geom='line',fun=median,color="black",lwd=1)+
  geom_hline(yintercept=0, color="red",linetype="dashed")+
  scale_color_manual(values=cbPalette)+
  scale_fill_manual(values=cbPalette)+
  ylab("nLL difference from SG")+
  coord_cartesian(ylim=c(-10, 25))+
  #ylim(c(-50,50))+
  xlab("Model")+
  theme_classic()+
  theme(legend.position = "none")
ggsave("./plots/e1_nLL_diff_zoom.pdf",width=3,height=1.5)



# ranktestBF <- function(x,y){
#   outsim<-signRankGibbsSampler(x, y, progBar = T) #sign rank gibbs sampler 
#   dense<- density(outsim$deltaSamples)
#   ddense <- with(dense, approxfun(x, y, rule=1))
#   ifelse(is.na(ddense(0)), denominator <- .Machine$double.xmin, denominator <- ddense(0)) #if no density at 0, then default to the smallest number
#   BF<-dcauchy(0, location = 0, scale = 1/sqrt(2), log = FALSE)/denominator
#   return(BF)
# }
# 
# comp1 <- ranktestBF(subset(data,model=="AS")$fit, subset(data, model=="DB")$fit)
# comp2 <- ranktestBF(subset(data,model=="AS")$fit, subset(data, model=="VS")$fit)
# comp3 <- ranktestBF(subset(data,model=="AS")$fit, subset(data, model=="SG")$fit)
# comp4 <- ranktestBF(subset(data,model=="DB")$fit, subset(data, model=="VS")$fit)
# comp5 <- ranktestBF(subset(data,model=="DB")$fit, subset(data, model=="SG")$fit)
# comp6 <- ranktestBF(subset(data,model=="VS")$fit, subset(data, model=="SG")$fit)
# 
# comp1 <- extractBF(ttestBF(subset(data,model=="AS")$fit, subset(data, model=="DB")$fit,paired = T))$bf
# comp2 <- extractBF(ttestBF(subset(data,model=="AS")$fit, subset(data, model=="VS")$fit,paired = T))$bf
# comp3 <- extractBF(ttestBF(subset(data,model=="AS")$fit, subset(data, model=="SG")$fit,paired = T))$bf
# comp4 <- extractBF(ttestBF(subset(data,model=="DB")$fit, subset(data, model=="VS")$fit,paired = T))$bf
# comp5 <- extractBF(ttestBF(subset(data,model=="DB")$fit, subset(data, model=="SG")$fit,paired = T))$bf
# comp6 <- extractBF(ttestBF(subset(data,model=="VS")$fit, subset(data, model=="SG")$fit,paired = T))$bf
# 
# comp1 <- wilcox.test(subset(data,model=="AS")$fit, subset(data, model=="DB")$fit,paired=T)
# comp2 <- wilcox.test(subset(data,model=="AS")$fit, subset(data, model=="VS")$fit,paired=T)$statistic
# comp3 <- wilcox.test(subset(data,model=="AS")$fit, subset(data, model=="SG")$fit,paired=T)$statistic
# comp4 <- wilcox.test(subset(data,model=="DB")$fit, subset(data, model=="VS")$fit,paired=T)$statistic
# comp5 <- wilcox.test(subset(data,model=="DB")$fit, subset(data, model=="SG")$fit,paired=T)$statistic
# comp6 <- wilcox.test(subset(data,model=="VS")$fit, subset(data, model=="SG")$fit,paired=T)$statistic
# 
# aaa <- wilcox_effsize(data,formula = fit ~ model, paired = T)
# (pred_HM <- ggplot(aaa, aes(x=group1,y=group2,fill=log(BF)))+
#     xlab("Model 1")+
#     ylab("Model 2")+
#     geom_tile()+
#     geom_text(aes(label = round(log(BF), 2),color = BF>1))+
#     scale_color_manual(guide = "none", values = c("white", "black"))+
#     scale_fill_viridis(name="log(Bayes Factor)")+ #,limits=c(0,1)
#     theme_classic()+
#     theme(axis.text.x = element_text(angle = 45,hjust = 1),aspect.ratio = 1)
# )
# 
# m1 <- c("AS","AS","AS","AS","DB","DB","DB","VS","VS","SG")
# m2 <- c("AS","DB","VS","SG","DB","VS","SG","VS","SG","SG")
# BF <- c(NA,comp1,comp2,comp3,NA,comp4,comp5,NA,comp6,NA)
# 
# BF_df <- data.frame(m1,m2,BF)
# BF_df$m1 <- factor(BF_df$m1,levels=c("AS","DB","VS","SG"))
# BF_df$m2 <- factor(BF_df$m2,levels=c("AS","DB","VS","SG"))
# 
# (pred_HM <- ggplot(BF_df, aes(x=m2,y=m1,fill=log(BF)))+
#     xlab("Model 1")+
#     ylab("Model 2")+
#     geom_tile()+
#     geom_text(aes(label = round(log(BF), 2),color = BF>1))+
#     scale_color_manual(guide = "none", values = c("white", "black"))+
#     scale_fill_viridis(name="log(Bayes Factor)")+ #,limits=c(0,1)
#     theme_classic()+
#     theme(axis.text.x = element_text(angle = 45,hjust = 1),aspect.ratio = 1)
# )
# (pred_acc_1 <- ggplot(data, aes(x=model,y=fit,color=model))+
#   geom_beeswarm(alpha = 0.3)+
#   geom_boxplot(outlier.shape = NA, width = 0.25,color="black")+
#   # stat_summary(fun=mean, color="black",geom="point",
#   #              shape=16, size=0.5,show_guide = FALSE)+ 
#   # stat_summary(fun.data = mean_cl_normal,
#   #              geom = "linerange",color="black") +
#   geom_hline(yintercept = -log(1/121)*8*14,color="red",linetype="dashed")+
#   theme_classic()+
#   scale_color_manual(values=cbPalette)+
#   ylab("nLL")+
#   xlab("Model")+
#   theme(legend.position="none")+
#   ggtitle("Experiment 1"))
# 
# data_bf <- read.csv("./Data/model_fits_e2_soc.csv")
# table(data_bf$model)
# 
# data <- read.csv("./Data/fit+pars_e1_nLL.csv")
# data$model <- factor(data$model,levels = c("AS","DB","VS","SG"))
# data[data$r2<0,"r2"]=0
# 
# ggplot(data, aes(x=model,y=r2,color=model))+
#   #geom_beeswarm(alpha=0.4)+
#   geom_boxplot()+
#   # stat_summary(fun.data = mean_cl_normal,
#   #              geom = "linerange",color="black") +
#   # stat_summary(fun=mean, color="black",geom="point",
#   #              shape=16, size=0.75,show_guide = FALSE)+ 
#   theme_classic()+
#   scale_color_manual(values=cbPalette)+
#   scale_fill_manual(values=cbPalette)


data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
model_perf_e2 <- data%>%group_by(model)%>%dplyr::summarize(pseudo_r2 = mean(r2))
data <- read.csv("./Data/fit+pars_e2_soc.csv")
nLL_e2 <- data%>%group_by(model)%>%dplyr::summarize(nLL_mean = mean(fit))
model_perf_e2$nLL_mean <- nLL_e2$nLL_mean

data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
data$model <- factor(data$model,levels = c("AS","DB","VS","SG"))

data <- subset(data,model=="AS")
data$fit_AS <- data$fit_AS/data$fit_SG
data$fit_DB <- data$fit_DB/data$fit_SG
data$fit_VS <- data$fit_VS/data$fit_SG
data$fit_SG <- data$fit_SG/data$fit_SG
data <- subset(data, select=-c(model))

data <- data%>%pivot_longer(cols=c(starts_with("fit_")),
                            names_to = "model",
                            values_to = "fit",
                            names_prefix = "fit_")
data$model <- factor(data$model,levels = c("AS","DB","VS","SG"))
data <- distinct(data)


(e2 <- ggplot(subset(data,model!="SG"),aes(x=model,y=fit,color=model,fill=model))+
  stat_summary(geom='point',fun=mean)+
  stat_summary(fun.data = mean_se, geom = "errorbar",width=0.3)+
  geom_hline(yintercept=1, color="red",linetype="dashed")+
  scale_color_manual(values=cbPalette,name="Model")+
  scale_fill_manual(values=cbPalette,name="Model")+
  ylab("nLL ratio model/SG")+
  xlab("Model")+
  ggtitle("Exp. 2  - Group rounds")+
  theme_classic()+
  theme(legend.position="none",legend.direction = "horizontal"))
ggsave("./plots/e2_nLL_diff_full.pdf",width=5,height=3.5)

ggplot(subset(data,model!="SG"),aes(x=model,y=fit,color=model))+
  geom_point()+
  geom_line(aes(group=id),alpha=0.5)+
  stat_summary(aes(group=1),geom='line',fun=median,color="black",lwd=1)+
  geom_hline(yintercept=0, color="red",linetype="dashed")+
  scale_color_manual(values=cbPalette)+
  scale_fill_manual(values=cbPalette)+
  ylab("nLL difference from SG")+
  coord_cartesian(ylim=c(-5, 10))+
  #ylim(c(-50,50))+
  xlab("Model")+
  theme_classic()+
  theme(legend.position = "none")
ggsave("./plots/e2_nLL_diff_zoom.pdf",width=5,height=3.5)

s5 <- cowplot::plot_grid(e1,e2,nrow=1,labels="auto")
ggsave("./plots/S5_dotrange_mean.pdf",height=3.5,width=6)

#######What is the right way to explore?###################
#AS only for optimal pars
data = read.csv("./Data/evoSim_ASonly.csv")
means <- data%>%group_by(gen)%>%dplyr::summarize(lambda = mean(lambda),beta=mean(beta),tau=mean(tau))
AS_lam <- ggplot(means,aes(x=gen,y = lambda))+
  geom_line(color="#999999")+
  theme_classic()+
  #ylab(expression(lambda))+
  ylab("Generalization")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
AS_final_lam <- subset(means,gen==499)$lambda

AS_bet <- ggplot(means,aes(x=gen,y = beta))+
  geom_line(color="#999999")+
  theme_classic()+
  #ylab("expression(beta)")+
  ylab("Directed exploration")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
AS_final_bet <- subset(means,gen==499)$beta

AS_tau <- ggplot(means,aes(x=gen,y = tau))+
  geom_line(color="#999999")+
  theme_classic()+
  #ylab(expression(tau))+
  ylab("Random exploration")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
AS_final_tau <- subset(means,gen==499)$tau

AS_pars <- cowplot::plot_grid(AS_lam,AS_bet,AS_tau,nrow=1,labels="auto")

#best eps_soc for pilot priors
data = read.csv("./Data/evoSim_SGepswpilot.csv")
means <- data%>%group_by(gen)%>%dplyr::summarize(lambda = mean(lambda),beta=mean(beta),tau=mean(tau),eps_soc=mean(eps_soc))
SG_lam <- ggplot(means,aes(x=gen,y = lambda))+
  geom_line(color="#56B4E9")+
  theme_classic()+
  #ylab(expression(lambda))+
  ylab("Generalization")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
SG_final_lam <- subset(means,gen==499)$lambda

SG_bet <- ggplot(means,aes(x=gen,y = beta))+
  geom_line(color="#56B4E9")+
  theme_classic()+
  ylab(expression(beta))+
  ylab("Directed exploration")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
SG_final_bet <- subset(means,gen==499)$beta

SG_tau <- ggplot(means,aes(x=gen,y = tau))+
  geom_line(color="#56B4E9")+
  theme_classic()+
  #ylab(expression(tau))+
  ylab("Random exploration")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
SG_final_tau <- subset(means,gen==499)$tau

SG_eps <- ggplot(means,aes(x=gen,y = eps_soc))+
  geom_line(color="#56B4E9")+
  theme_classic()+
  #ylab(expression(epsilon[soc]))+
  ylab("Social noise")+ 
  theme(axis.title.y = element_text(angle = 0,vjust=0.5,hjust=0.5),legend.direction="horizontal",legend.position = "None")+
  xlab("Generation")
SG_final_eps <- subset(means,gen==499)$eps_soc


SG_pars <- cowplot::plot_grid(SG_lam,SG_bet,SG_tau,SG_eps,nrow=1,labels="auto")

title <- ggdraw() + 
  draw_label(
    "Parameter evolution with baseline parameters bounded in pilot ranges",
    fontface = 'bold',
    x = 0,
    hjust = 0
  ) +
  theme(
    plot.margin = margin(0, 0, 0, 7)
  )
SG_pars <- cowplot::plot_grid(title, SG_pars, ncol = 1, rel_heights = c(0.1, 1))
ggsave("./plots/supp_SGpilotbesteps.pdf",width=9,height=3,plot=SG_pars)

#parameter cutoffs for evoSim
data1 <- read.csv("./Data/fit+pars_poisson_nLL_mean.csv")
data2 <- read.csv("./Data/fit+pars_soc_poisson_nLL_mean.csv")
data <- rbind(data1,data2)
quantile(data$lambda,c(0.4,0.6))
quantile(data$beta,c(0.4,0.6))
quantile(data$tau,c(0.4,0.6))
quantile(subset(data,model=="SG")$par)

data <- read.csv("./Data/pilot_sim_SG_eps_soc_varied.csv")
ggplot(data,aes(x=eps_soc,y=reward))+
  stat_summary(mapping=aes(x=round(eps_soc*20)/20,y=reward),fun=mean,geom='point',alpha=0.5)+
  #geom_point()+
  geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
  theme_classic()+
  ylab("Reward")+
  #xlab(expression(epsilon[soc]))+
  xlab("Social noise")
  ggtitle(paste0("Effect of social noise values on reward\nin participant parameter space"))+
  theme(plot.title = element_text(size=12))
ggsave("./plots/eps_soc_rew_range.pdf") #this contradicts my correlation though?


###########
#Model variants
###########
#VS agnostic
data <- read.csv("./Data/VS_payoff.csv")
data <- subset(data,model!="AS")
data$value <- ifelse(data$model=="VS_mem","Yes","No")
data$value <- factor(data$value)
cols <- c("#66CC99","#009E73")
(VS_comp <- ggplot(data,aes(x=value,y=reward,color=value,fill=value))+
  stat_summary(fun=mean,geom="bar")+
  stat_summary(fun.data = mean_ci,geom="errorbar",color="black",linewidth=0.5,width=0.25)+
  theme_classic()+
  scale_color_manual(values=cols)+
  scale_fill_manual(values=cols)+
  xlab("Value-sensitive")+
  ylab("Reward")+
  ggtitle("VS variants")+
  coord_cartesian(ylim = c(0.5,0.57))+
  theme(legend.position="none"))
#ggsave("./plots/VS_payoff_comp.pdf",height=5,width=5)
meaned <- data%>%group_by(model,group,agent)%>%dplyr::summarize(rew = mean(reward))
ttestPretty(subset(meaned,model=="VS_mem")$rew,subset(meaned,model=="VS_agnostic")$rew)


#DB payoff
data <- read.csv("./Data/DB_payoff.csv")
data <- subset(data,model!="AS")
data$value <- ifelse(data$model=="DB_val","Yes","No")
data$value <- factor(data$value)
cols <- c("#E69F00","#FF964F")
DB_comp <- ggplot(data,aes(x=value,y=reward,color=value,fill=value))+
  stat_summary(fun=mean,geom="bar")+
  stat_summary(fun.data = mean_ci,geom="errorbar",color="black",linewidth=0.5,width=0.25)+
  theme_classic()+
  scale_color_manual(values=cols)+
  scale_fill_manual(values=cols)+
  xlab("Value-sensitive")+
  ylab("Reward")+
  ggtitle("DB variants")+
  coord_cartesian(ylim = c(0.5,0.55))+
  theme(legend.position="none")
#ggsave("./plots/DB_payoff_comp.pdf",height=5,width=5)
meaned <- data%>%group_by(model,group,agent)%>%dplyr::summarize(rew = mean(reward))
ttestPretty(subset(meaned,model=="DB")$rew,subset(meaned,model=="DB_val")$rew)

#SG indiscriminate
data <- read.csv("./Data/SG_vs_dummy.csv")
data <- subset(data,model!="AS")
data$disc <- ifelse(data$model=="dummy","No","Yes")
data$disc <- factor(data$disc)
cols <- c("#67B1C0","#56B4E9")
(SG_comp <- ggplot(data,aes(x=disc,y=reward,color=disc,fill=disc))+
  stat_summary(fun=mean,geom="bar")+
  stat_summary(fun.data = mean_ci,geom="errorbar",color="black",linewidth=0.5,width=0.25)+
  theme_classic()+
  scale_color_manual(values=cols)+
  scale_fill_manual(values=cols)+
  xlab("Discriminate social info use")+
  ylab("Reward")+
  ggtitle("SG variants")+
  coord_cartesian(ylim=c(0.5,0.6))+
  theme(legend.position="none"))
#ggsave("./plots/SG_comp.pdf",height=5,width=5)
meaned <- data%>%group_by(model,group,agent)%>%dplyr::summarize(rew = mean(reward))
ttestPretty(subset(meaned,model=="SG")$rew,subset(meaned,model=="dummy")$rew)


model_vars <- cowplot::plot_grid(VS_comp,DB_comp,SG_comp,nrow=1,labels="auto")
ggsave("./plots/supp_model_vars.pdf",width=8,height=2.5)

######################
#exclusive analysis (no social noise at bound)
######################

data <- read.csv("./Data/fit+pars_e1_nLL.csv")

pilot_data <- read.csv("./Data/e1_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]

pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()
randomChoicePerc <- mean(subset(pilot_data,trial!=0)$isRandom)
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(agent,group)%>%dplyr::summarize(meanReward =mean(reward),soc_sd=mean(soc_sd,na.rm=T))
data <- merge(meandata,data,by=c("agent","group"))
data$SG_best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$SG_best <- factor(data$SG_best)

#reward over eps soc
eps_rew_test <- cor.test(subset(data,model=="SG"&SG_best==1&data$par<18.9999)$par,subset(data,model=="SG"&SG_best==1&data$par<18.9999)$meanReward,method = "kendall")
label = c(paste0("r[tau]==",round(eps_rew_test$estimate,2)),paste0("p==",round(eps_rew_test$p.value,3)))
(eps_soc_rew_e1 <- ggplot(subset(data,data$model=="SG"&data$SG_best==1&data$par<18.9999),aes(x=par,y=meanReward))+
    geom_point(color="black")+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    theme_classic()+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    ylab("Mean Reward")+
    ggtitle("Exp. 1")+
    theme(legend.position = "none")+
    annotate("text",x=5,y=c(0.715,0.7),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1&par<18.999)$par,subset(data,model=="SG"&SG_best==1&par<18.999)$meanReward,method = "kendall")


#beta over eps soc
beta_expl_test <- cor.test(subset(data,model=="SG"&SG_best==1&par<18.999)$par,subset(data,model=="SG"&SG_best==1&par<18.999)$beta,method = "kendall")
label = c(paste0("r[tau]==",round(beta_expl_test$estimate,2)),paste0("p==",round(beta_expl_test$p.value,3)))

(expl_replace_e1 <- ggplot(subset(data, model=="SG"&SG_best==1&par<18.999),aes(x=par,y=beta))+
    geom_point()+
    theme_classic()+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    ggtitle("Exp. 1")+
    #ylab(expression(beta))+
    ylab("Directed exploration")+
    annotate("text",x=5,y=c(0.55,0.51),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1&par<18.999)$par,subset(data,model=="SG"&SG_best==1&par<18.999)$beta,method = "kendall")

#E2

data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")

pilot_data <- read.csv("./Data/e2_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]

randomChoicePerc <- mean(subset(pilot_data,trial!=0)$isRandom)
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(agent,group,taskType)%>%dplyr::summarize(meanReward =mean(reward),soc_sd=mean(soc_sd,na.rm=T))
data <- merge(subset(meandata,taskType=="social"),data,by=c("agent","group"))
data$SG_best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$SG_best <- factor(data$SG_best)

#reward over eps soc
eps_rew_test <- cor.test(subset(data,model=="SG"&SG_best==1&par<18.9999)$par,subset(data,model=="SG"&SG_best==1&par<18.9999)$meanReward,method = "kendall")
label = c(paste0("r[tau]==",round(eps_rew_test$estimate,2)),paste0("p==",round(eps_rew_test$p.value,3)))
(eps_soc_rew_e2 <- ggplot(subset(data,data$model=="SG"&data$SG_best==1&data$par<18.9999),aes(x=par,y=meanReward))+
    geom_point(color="black")+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    theme_classic()+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    ylab("Mean Reward")+
    ggtitle("Exp. 2")+
    theme(legend.position = c(0.625,0.83))+
    annotate("text",x=3.5,y=c(0.73,0.715),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1&data$par<18.9999)$par,subset(data,model=="SG"&SG_best==1&data$par<18.9999)$meanReward,method = "kendall")

#beta over eps soc
beta_expl_test <- cor.test(subset(data,model=="SG"&SG_best==1&par<18.9999)$par,subset(data,model=="SG"&SG_best==1&par<18.9999)$beta,method = "kendall")
label = c(paste0("r[tau]==",round(beta_expl_test$estimate,2)),paste0("p==",round(beta_expl_test$p.value,4)))

(expl_replace_e2 <- ggplot(subset(data, model=="SG"&SG_best==1&par<18.9999),aes(x=par,y=beta))+
    geom_point()+
    theme_classic()+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    #ylab(expression(beta))+
    ylab("Directed exploration")+
    ggtitle("Exp. 2")+
    annotate("text",x=3.5,y=c(0.45,0.415),label = label,parse=T))


corTestPretty(subset(data,model=="SG"&SG_best==1&par<18.9999)$par,subset(data,model=="SG"&SG_best==1&par<18.9999)$beta,method = "kendall")

excl_analysis <- cowplot::plot_grid(eps_soc_rew_e1,expl_replace_e1,eps_soc_rew_e2,expl_replace_e2,
                                         nrow=2,labels="auto")
ggsave("./plots/supp_excl_analysis.pdf",plot=excl_analysis,width=7,height=6)


#######
#social learning makes you a better learner
#######
data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
e1_data <- read.csv("./Data/fit+pars_e1_nLL.csv")
group_lambda <- mean(rbind(subset(data,model=="SG"),subset(e1_data,model=="SG"))$lambda)
group_beta <- mean(rbind(subset(data,model=="SG"),subset(e1_data,model=="SG"))$beta)
data$best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$best <- factor(data$best)
data$taskType <- "Group"


data_ind <- read.csv("./Data/fit+pars_e2_ind_nLL.csv")
ind_lambda <- mean(subset(data_ind,model=="AS")$lambda)
ind_beta <- mean(subset(data_ind,model=="AS")$beta)
data_ind$best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_AS"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data_ind$best <- factor(data_ind$best)
data_ind$taskType <- "Solo"

parcomp = rbind(subset(data,model=="SG"),subset(data_ind,model=="AS"))
parcomp$model <- factor(parcomp$model,levels=c("AS","SG"))
(betacomp <- ggplot(parcomp,aes(x=model,y=beta,color=model,fill=model))+
    geom_line(aes(group=id),alpha=0.3)+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    geom_signif(
      comparisons = list(c("AS", "SG")),
      map_signif_level = T,color="black",
      test = "wilcox.test",test.args = list(paired=TRUE))+
    scale_colour_manual(values=c("#999999","#56B4E9"))+
    theme_classic()+
    scale_x_discrete(labels=c("AS" = "AS\n(solo rounds)", "SG" = "SG\n(group rounds)"))+
    xlab("")+
    #ylab(expression(epsilon["soc"]))+
    #ylab(expression(beta))+
    ylab("Directed exploration")+
    theme(legend.position = "None"))

ranktestPretty(subset(parcomp,model=="AS")$beta,subset(parcomp,model=="SG")$beta,paired=T)

(lambdacomp <- ggplot(parcomp,aes(x=model,y=lambda,color=model,fill=model))+
    geom_line(aes(group=id),alpha=0.3)+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    geom_signif(
      comparisons = list(c("AS", "SG")),
      map_signif_level = T,color="black",
      test = "wilcox.test",test.args = list(paired=TRUE))+
    scale_colour_manual(values=c("#999999","#56B4E9"))+
    theme_classic()+
    scale_x_discrete(labels=c("AS" = "AS\n(solo rounds)", "SG" = "SG\n(group rounds)"))+
    xlab("")+
    #ylab(expression(epsilon["soc"]))+
    #ylab(expression(lambda))+
    ylab("Generalization")+
    theme(legend.position = "None"))

ranktestPretty(subset(parcomp,model=="AS")$lambda,subset(parcomp,model=="SG")$lambda,paired=T)

#simulated best beta
data <- read.csv("./Data/GP_fitpars_betasim.csv")
prev_lit_pars <- read.csv("./Data/Wu_modelFit.csv")
prev_lit_pars <- subset(prev_lit_pars,reward=="Cumulative"&kernel=="Function Learning"&acq=="UCB"&environment=="Smooth")
prev_beta <- mean(prev_lit_pars$beta)
prev_lambda <- mean(prev_lit_pars$lambda)
beta_good <- lm(reward~beta*model,data=data)
summary(beta_good)

linelist <- c("solid","dashed")
socAsoc <- c("#999999", "#0072B2")
(beta_rew <- ggplot(data,aes(x=round(beta*10)/10,y=reward,color=model, fill=model))+
    stat_summary(fun.data=mean_cl_boot,geom='ribbon',alpha=0.5,color=NA)+
    stat_summary(fun=mean,geom='line')+
    geom_vline(aes(linetype="Wu et al. (2018)",xintercept = prev_beta),color="#999999")+
    geom_vline(aes(linetype="Parameter estimate",xintercept = group_beta),color="#0072B2")+
    geom_vline(aes(linetype="Parameter estimate",xintercept = ind_beta),color="#999999")+
    scale_linetype_manual(values=linelist,name=" ")+
    theme_classic()+
    scale_color_manual(values=socAsoc,name="Model")+
    scale_fill_manual(values=socAsoc,name="Model")+
    theme(legend.position = c(0.75,0.7),plot.title = element_text(size = 12),
          strip.background=element_blank(),legend.background=element_blank(),
          legend.key=element_blank())+
    xlab(expression(paste(beta," (Directed exploration)")))+
    xlab("Directed exploration")+ 
    ggtitle("Effect of directed exploration on reward in simulations")+
    ylab("Reward"))

data <- read.csv("./Data/GP_fitpars_lambdasim.csv")

socAsoc <- c("#999999", "#0072B2")
(lam_rew <- ggplot(data,aes(x=round(lambda*10)/10,y=reward,color=model, fill=model))+
    stat_summary(fun.data=mean_cl_boot,geom='ribbon',alpha=0.5,color=NA)+
    stat_summary(fun=mean,geom='line')+
    geom_vline(aes(linetype="Wu et al. (2018)",xintercept = prev_lambda),color="#999999")+
    geom_vline(aes(linetype="Parameter estimate",xintercept = group_lambda),color="#0072B2")+
    geom_vline(aes(linetype="Parameter estimate",xintercept = ind_lambda),color="#999999")+
    scale_linetype_manual(values=linelist,name=" ")+    #geom_smooth()+ #method="loess",span=5
    theme_classic()+
    scale_color_manual(values=socAsoc,name="Model")+
    scale_fill_manual(values=socAsoc,name="Model")+
    theme(legend.position = c(0.7,0.3),plot.title = element_text(size = 12),
          strip.background=element_blank(),legend.background=element_blank(),
          legend.key=element_blank())+
    #xlab(expression(lambda))+
    xlab(expression(paste(lambda, " (Generalization)")))+ 
    #ylim(c(0.5,1))+
    ggtitle("Effect of generalization on reward in simulations")+
    ylab("Reward"))

(soc_good <- cowplot::plot_grid(betacomp,beta_rew,lambdacomp,lam_rew,nrow = 2, labels="auto"))
ggsave("./plots/soc_good.pdf",plot=soc_good,width=9.1,height=6.5)


#parameters for SG best fit only e2
data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
data$model <- factor(data$model,levels=c("AS","DB","VS","SG"))

pilot_data <- read.csv("./Data/e2_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(agent,group,taskType)%>%dplyr::summarize(meanReward =mean(reward),soc_sd=mean(soc_sd,na.rm=T))
data <- merge(subset(meandata,taskType=="social"),data,by=c("agent","group"))
data$SG_best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$SG_best <- factor(data$SG_best)
(lam <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=lambda,color="#56B4E9"))+
    #scale_y_log10()+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    scale_colour_manual(values="#56B4E9")+
    theme_classic()+
    xlab("Generalization")+
    #ylab(expression(lambda))+
    #ylab("Generalization")+
    ylab("Parameter value")+
    theme(legend.position = "None",axis.text.x = element_blank(),axis.ticks.x = element_blank()))

avg_lamb <- mean(subset(data,model=="SG"&SG_best==1)$lambda)
ttestPretty(subset(data,model=="SG"&SG_best==1)$lambda,mu=2)


(bet <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=beta,color="#56B4E9"))+
    #scale_y_log10()+
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


(tau <- ggplot(subset(data,model=="SG"&SG_best==1),aes(x=model,y=tau,color="#56B4E9"))+
    #scale_y_log10()+
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

(eps_soc <- ggplot(subset(data,data$model=="SG"&SG_best==1),aes(x=model,y=par,color="#56B4E9"))+
    #scale_y_log10()+
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
pars <- ggdraw(add_sub(pars, "SG Parameters", vpadding=grid::unit(0,"lines"),y=6, x=0.55, vjust=4.5))
ggsave("./plots/pars_E2.pdf",width=7.5,height=5)

#####
#At what correlations does SG break?
#####

#no correlation
data = read.csv("./Data/evoSim_corr00.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.0
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- subset(counts,gen==499)
rm(data)

#corr=0.1
data = read.csv("./Data/evoSim_corr01.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.1
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.2
data = read.csv("./Data/evoSim_corr02.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.2
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.3
data = read.csv("./Data/evoSim_corr03.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.3
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.4
data = read.csv("./Data/evoSim_corr04.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.4
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.5
data = read.csv("./Data/evoSim_corr05.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.5
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.6
data = read.csv("./Data/evoSim.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.6
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.7
data = read.csv("./Data/evoSim_corr07.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.7
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.8
data = read.csv("./Data/evoSim_corr08.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.8
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

#corr=0.9
data = read.csv("./Data/evoSim_corr09.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
data$mix =  factor(data$mix,levels=c("AS","DB","VS","SG","AS.DB","AS.VS","AS.SG","DB.VS","DB.SG","VS.SG",
                                     "AS.DB.VS","AS.DB.SG","AS.VS.SG","DB.VS.SG","AS.DB.VS.SG"))
data$corr <- 0.9
counts <- data%>%group_by(gen,mix,model,corr)%>%dplyr::summarize(n=n())
win_by_cor <- rbind(win_by_cor,subset(counts,gen==499))
rm(data)

ggplot(win_by_cor,aes(x=corr,y=n/1000,color=model,fill=model))+
  stat_summary(geom = 'ribbon', fun.data = mean_cl_boot,alpha=0.3,color=NA)+
  stat_summary(geom='line',fun=mean)+
  theme_classic()+
  scale_color_manual(values = cbPalette,name='Model')+ 
  scale_fill_manual(values = cbPalette,name='Model')+ 
  ylab('P(model) in final gen.')+
  xlab('Environment correlation')+
  theme(legend.position = c(0.9,0.5))
ggsave('./plots/evoSim_winner_by_corr.pdf',width = 5,height = 2.5)
