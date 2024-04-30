#Exp 2 analysis (mostly just stealing my own full analysis code teehee)
#Alex, very jetlagged from an airport in 2023

library(ggplot2)
library(dplyr)
library(ggtern)
library(viridis)
library(tidyr)
library(cowplot)
library(lme4)
library(lmerTest)
library(ggbeeswarm)

library(ggsignif)
library(viridis)
library(brms)
library(sjPlot)

saveAll <- F
cbPalette <- c("#999999","#E69F00", "#009E73","#56B4E9", "#CC79A7", "#F0E442", "#0072B2", "#D55E00")#

####extra imports and functions #######
source('statisticalTests.R')
std <- function(a) sd(a,na.rm=T) / sqrt(sum(!is.na(a)))

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


#Fig 4 - behaviour ---------------------------------
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9",  "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
saveAll <- F

#demography #########
demo <- read.csv("./Data/e2_demo.csv")
meanAge <- mean(as.numeric(demo$Age),na.rm=T)
sdAge <- sd(as.numeric(demo$Age),na.rm=T)

meanCompletion <- mean(demo$Time.taken)/60
semCompletion <- std(demo$Time.taken)/60

meanPayout <- mean(demo$totalPayment,na.rm=T)
semPayout <- std(demo$totalPayment)

table(demo$Sex)

#data imports + sanity check LCs + coherence##########
pilot_data <- read.csv("./Data/e2_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]
pilot_data$id <-  pilot_data %>% group_by(agent,group) %>% group_indices()

randomChoicePerc <- mean(subset(pilot_data,trial!=0)$isRandom)
pilot_data$taskType <- factor(pilot_data$taskType)

socAsoc <- c("#999999", "#1F78B4","red")#, "#D55E00" "#999999"

meandata <- pilot_data%>%group_by(group,trial,taskType)%>%summarise(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                           search_dist=mean(search_dist),variance=mean(variance),
                                                           soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                           soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))

#learning curves
(lc <- ggplot(meandata,aes(x=trial,y=meanReward,linetype=taskType,color=taskType))+
    geom_line(aes(group=interaction(group,taskType)),alpha=0.15)+ #color="black
    geom_hline(yintercept=0.5, linetype="dashed",color="red",lwd=1)+
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = 0.3, mapping=aes(fill=taskType),color = NA)+
    stat_summary(fun=mean,geom="line",mapping=aes(color=taskType),lwd=1)+
    theme_classic()+
    scale_color_manual(values = socAsoc, labels=c("Solo","Group")) +
    scale_fill_manual(values = socAsoc, labels=c("Solo","Group")) +
    scale_linetype(name = "Task type", labels=c("Solo","Group"))+
    labs(color="Task type", fill = "Task type",linetype="Task type") +
    theme(legend.position=c(0.15,0.85), strip.background=element_blank(), 
          legend.background=element_blank(), legend.key=element_blank())+
    ylab("Average Reward")+
    xlab("Trial"))

#inset plot, rew difference within subject
rewdiff <- pilot_data %>%
  group_by(id,taskType) %>%
  summarise(meanReward = mean(reward)) %>%
  pivot_wider(names_from = taskType, values_from = meanReward) %>%
  mutate(diff = social - individual) %>%
  ungroup()
ggplot(rewdiff,aes(x=diff))+
  geom_histogram(bins=20)+
  geom_vline(xintercept = 0,color="red",lwd=1.5,linetype="dashed")+
  theme_classic()+
  xlab("Reward difference\ngroup - solo")+
  ylab("Count")

perc_better_soc <- sum(rewdiff$diff>0)/nrow(rewdiff)

#social search distances over time
(soc_sd_trial <- ggplot(meandata,aes(x=trial,y=soc_sd,linetype=taskType,color=taskType))+
    geom_line(aes(group=interaction(group,taskType)),alpha=0.15)+ #color="black
    stat_summary(fun.data = mean_cl_boot, geom = 'ribbon', alpha = 0.3, mapping=aes(fill=taskType),color = NA)+
    stat_summary(fun=mean,geom="line",mapping=aes(color=taskType),lwd=1)+
    theme_classic()+
    scale_color_manual(values = socAsoc, labels=c("Solo","Group")) +
    scale_fill_manual(values = socAsoc, labels=c("Solo","Group")) +
    scale_linetype(name = "Task type", labels=c("Solo","Group"))+
    labs(color="Task type", fill = "Task type",linetype="Task type") +
    theme(legend.position=c(0.15,0.15), strip.background=element_blank(), 
          legend.background=element_blank(), legend.key=element_blank())+
    ylab("Social search distance")+
    xlab("Trial"))

#exclude random choices for analysis
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(id,taskType)%>%summarise(meanReward =mean(reward),coherence=mean(coherence), prev_rew = mean(prev_rew),
                                                                    search_dist=mean(search_dist),variance=mean(variance),
                                                                    soc_sd1 = mean(soc_sd1), soc_sd2 = mean(soc_sd2), soc_sd3 = mean(soc_sd3), soc_sd = mean(soc_sd),
                                                                    soc_rew1 = mean(soc_rew1), soc_rew2 = mean(soc_rew2),soc_rew3 = mean(soc_rew3),soc_rew = mean(soc_rew))


#higher rewards in group rounds
ttestPretty(subset(meandata,taskType=="social")$meanReward,subset(meandata,taskType=="individual")$meanReward,paired=T)
#lower search distances in froup rounds
ttestPretty(subset(meandata,taskType=="social")$soc_sd,subset(meandata,taskType=="individual")$soc_sd,paired=T)


# Social search distance ~ previous reward ##############
#Human Data
data = read.csv("./Data/e2_data_regressable.csv")
data$social = factor(data$social)
data$taskType <- factor(data$taskType)
data <- subset(data,isRandom==0)

dist_prev_rew = run_model(brm(soc_sd ~ soc_rew * social * taskType + (1 + soc_rew + social + taskType|group/agent),
                              data = data, cores = 4, iter = 4000, warmup = 1000,
                              control = list(adapt_delta = 0.99,max_treedepth = 20)), modelName = 'dist_prev_rew_e2')

#posterior samples
post <- dist_prev_rew %>% brms::posterior_samples() 

formatHDI(post$b_soc_rew+post$b_taskTypesocial+post$`b_soc_rew:taskTypesocial`) #individual previous reward in group rounds
formatHDI(post$b_soc_rew+post$b_social1+post$b_taskTypesocial+post$`b_soc_rew:social1`+
          post$`b_soc_rew:taskTypesocial`+post$`b_social1:taskTypesocial`+
          post$`b_soc_rew:social1:taskTypesocial`) #social previous reward in group rounds

formatHDI(post$b_social1+post$`b_soc_rew:social1`+post$`b_social1:taskTypesocial`+ post$`b_soc_rew:social1:taskTypesocial`) #contrast social vs individual previous reward in group rounds


formatHDI(post$b_soc_rew) #individual previous reward in solo rounds
formatHDI(post$b_soc_rew+post$b_taskTypesocial+post$`b_soc_rew:taskTypesocial`) #contrast individual previous reward group vs. solo
formatHDI(post$b_soc_rew+post$b_social1+post$`b_soc_rew:social1`) #social previous reward in solo rounds
formatHDI(post$b_taskTypesocial+ post$`b_soc_rew:taskTypesocial`+post$`b_social1:taskTypesocial`+post$`b_soc_rew:social1:taskTypesocial`) #contrast social previous reward group vs. solo

#format data for predictions
prev_rew = seq(0,50)/50
prev_rew = prev_rew[2:length(prev_rew)]
test <- expand.grid(soc_rew = prev_rew,social=levels(data$social),taskType=levels(data$taskType))
preds = fitted(dist_prev_rew, re_formula=NA,resp=c("soc_rew","social","taskType"),newdata=test,probs=c(.025,.975))
plotdata = data.frame(prev_rew=test$soc_rew,social=test$social,taskType=test$taskType,sdist=preds[,1],se=preds[,2],lower=preds[,3],upper=preds[,4])

#get interaction of task and info type as a variable

plotdata$int <- interaction(factor(plotdata$social),plotdata$taskType)
plotdata$int <- recode_factor(plotdata$int, "0.individual" = "Individual + solo",
                              "1.individual" = "Social + solo", "0.social"= "Individual + group",
                              "1.social" = "Social + group")
plotdata$int <- factor(plotdata$int, levels=c("Individual + solo","Individual + group","Social + solo", "Social + group"))

data$int <- interaction(factor(data$social),data$taskType)
data$int <- recode_factor(data$int, "0.individual" = "Individual + solo",
                          "1.individual" = "Social + solo", "0.social"= "Individual + group",
                          "1.social" = "Social + group")
data$int <- factor(data$int, levels=c("Individual + solo","Individual + group","Social + solo", "Social + group"))

exp2pal <- c("#999999","#000000","#A6CEE3","#1F78B4")

(p <- ggplot()+
    stat_summary(data,mapping=aes(round(soc_rew*50)/50,y=soc_sd,color=int,fill=int),fun=mean,geom='point',alpha=0.8)+
    geom_line(plotdata,mapping=aes(x=prev_rew,y=sdist,color=int),lwd=1.25)+
    geom_ribbon(plotdata,mapping=aes(x=prev_rew,y=sdist,ymin=lower,ymax=upper,fill=int),alpha=0.3)+
    scale_color_manual(values=exp2pal,name="Info source + task type")+
    scale_fill_manual(values=exp2pal,name="Info source + task type")+
    theme_classic()+
    xlab("Previous Reward")+
    ylab("Search distance")+
    theme(legend.position = c(0.65,0.8), legend.background = element_blank(), legend.key = element_blank()))
a <- addSmallLegend(p,8,10,1)

#Social search distance####
data$dist <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"soc_sd"]==0,"0",ifelse(data[x,"soc_sd"]==1,"1",ifelse(data[x,"soc_sd"]<3,"2",">=3"))))
data$dist <- factor(data$dist,levels=c("0","1","2",">=3"))

data$rew_bins <- round(data$soc_rew*50)/50
pdata_grp <- subset(data,social==1&!is.na(soc_rew)&taskType=="social") #social==1&dist=='0'|social==1&dist=='1'

freqs <- pdata_grp%>%group_by(rew_bins,dist)%>%dplyr::summarize(n=n(),dist=unique(dist))%>%mutate(freq=n/sum(n))
freqs$dist <- factor(freqs$dist,levels=c("0","1","2",">=3"))
freqs_group <- freqs

pdata_ind <- subset(data,social==1&!is.na(soc_rew)&taskType=="individual") #social==1&dist=='0'|social==1&dist=='1'

freqs <- pdata_ind%>%group_by(rew_bins,dist)%>%dplyr::summarize(n=n(),dist=unique(dist))%>%mutate(freq=n/sum(n))
freqs$dist <- factor(freqs$dist,levels=c("0","1","2",">=3"))

freqs$taskType <-  "Solo"
freqs_group$taskType <- "Group"
freqs <- rbind(freqs_group,freqs)
freqs$taskType <- factor(freqs$taskType,levels=c("Solo","Group"))
#colors <- c("#fdbf6f", "#ff7f00", "#bdd5ea", "#4d88e0")
colors <- c("#fdbf6f", "#ff7f00", "#8da0cb","#425e8a")
freqs$int <- interaction(factor(freqs$dist),freqs$taskType)
freqs$int <- recode_factor(freqs$int, "0.Solo" = "0 + solo",
                          "1.Solo" = "1 + solo", "0.Group"= "0 + group",
                          "1.Group" = "1 + group")
freqs$int <- factor(freqs$int, levels=c("0 + solo","0 + group","1 + solo", "1 + group"))

(soc_sd <- ggplot(subset(freqs,dist=="0"|dist=="1"),aes(x=rew_bins,y=freq,color=int,fill=int))+ # subset(freqs,dist!=">=3"),linetype=taskType
    geom_point(alpha=0.8)+
    geom_smooth(method="lm")+
    theme_classic()+
    ylab("P(search distance)")+
    xlab("Previous social reward")+
    theme(legend.position = c(0.35,0.65),legend.background = element_blank(),
          legend.key = element_blank())+
    scale_fill_manual(values = colors, name="Search dist. + task type")+
    scale_color_manual(values = colors, name="Search dist. + task type")) #,end=.8

soc_sd_test <- lm(freq~rew_bins*dist*taskType,data=subset(freqs,dist=="0"|dist=="1"))
summary(soc_sd_test)

################################
#Fitting analysis
################################
#sanity check (pxp on solo rounds)
data = read.csv("./Data/pxp_e2_ind.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9", "#CC79A7",  "#F0E442", "#0072B2", "#D55E00")

(pxp_ind <- ggplot(data,aes(x=model,y=exceedance,color=model,fill=model))+
    geom_bar(stat="identity")+
    geom_hline(yintercept=0.25,linetype="dashed",color="red",lwd=1)+
    theme_classic()+
    scale_colour_manual(values=cbPalette)+
    scale_fill_manual(values=cbPalette)+
    xlab("Model")+
    ylab("pxp")+
    ggtitle("Solo rounds")+
    theme(legend.position="None"))

#protected exceedance probability
data = read.csv("./Data/pxp_e2_soc.csv")
data$model = factor(data$model,levels=c("AS","DB","VS","SG"))
cbPalette <- c("#999999", "#E69F00", "#009E73","#56B4E9", "#CC79A7",  "#F0E442", "#0072B2", "#D55E00")

(pxp_soc <- ggplot(data,aes(x=model,y=exceedance,color=model,fill=model))+
    geom_bar(stat="identity")+
    geom_hline(yintercept=0.25,linetype="dashed",color="red",lwd=1)+
    theme_classic()+
    scale_colour_manual(values=cbPalette)+
    scale_fill_manual(values=cbPalette)+
    xlab("Model")+
    ylab("pxp")+
    ggtitle("Group rounds")+
    theme(legend.position="None"))


#comparing beta between solo and group rounds
data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
data$best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$best <- factor(data$best)
data$taskType <- "Group"

data_ind <- read.csv("./Data/fit+pars_e2_ind_nLL.csv")
data_ind$best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_AS"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data_ind$best <- factor(data_ind$best)
data_ind$taskType <- "Solo"


#compare SG and AS generally
parcomp = rbind(subset(data,model=="SG"),subset(data_ind,model=="AS"))
parcomp$model <- factor(parcomp$model,levels=c("AS","SG"))

betap = wilcox.test(subset(parcomp,model=="AS")$beta,subset(parcomp,model=="SG")$beta,paired=T)$p.value

(betacomp <- ggplot(parcomp,aes(x=model,y=beta,color=model,fill=model))+
    geom_line(aes(group=id),alpha=0.3)+
    geom_beeswarm(cex=2.5,alpha=0.5)+
    stat_summary(fun.data = mean_cl_normal,  
                 geom = "linerange",color="black") +
    stat_summary(fun=mean, color="black",geom="point",
                 shape=16, size=1,show_guide = FALSE)+ 
    
    geom_signif(
      comparisons = list(c("AS", "SG")),
      color="black", annotations = paste0("p = ",sub("^0+", "", round(betap,3))),
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

#reward over eps soc
data <- read.csv("./Data/fit+pars_e2_soc_nLL.csv")
data$model <- factor(data$model,levels=c("AS","DB","VS","SG"))

pilot_data <- read.csv("./Data/e2_data.csv")
pilot_data = pilot_data[order(pilot_data$agent,pilot_data$group,pilot_data$round,pilot_data$trial),]
pilot_data <-  subset(pilot_data,trial!=0 & isRandom==0)

meandata <- pilot_data%>%group_by(agent,group,taskType)%>%summarise(meanReward =mean(reward),soc_sd=mean(soc_sd,na.rm=T))
data <- merge(subset(meandata,taskType=="social"),data,by=c("agent","group"))
data$SG_best <- sapply(1:dim(data)[1], function(x) ifelse(data[x,"fit_SG"]==min(data[x,c("fit_AS","fit_DB","fit_VS","fit_SG")]),1,0))
data$SG_best <- factor(data$SG_best)

eps_rew_test <- cor.test(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$meanReward,method = "kendall")
label = c(paste0("r[tau]==",round(eps_rew_test$estimate,2)),paste0("p==",round(eps_rew_test$p.value,3)))
(eps_soc_rew <- ggplot(subset(data,data$model=="SG"&data$SG_best==1),aes(x=par,y=meanReward))+
    geom_point(color="black")+
    geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
    theme_classic()+
    #xlab(expression(epsilon["soc"]))+
    xlab("Social noise")+
    ylab("Mean Reward")+
    theme(legend.position = c(0.625,0.83))+
    annotate("text",x=3.5,y=c(0.73,0.715),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$meanReward,method = "kendall")


#beta over eps soc
beta_expl_test <- cor.test(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$beta,method = "kendall")
label = c(paste0("r[tau]==",round(beta_expl_test$estimate,2)),paste0("p==",round(beta_expl_test$p.value,4)))

(expl_replace <- ggplot(subset(data, model=="SG"&SG_best==1),aes(x=par,y=beta))+
  geom_point()+
  theme_classic()+
  geom_smooth(method="lm",color="#56B4E9",fill="#56B4E9")+
  #xlab(expression(epsilon["soc"]))+
  xlab("Social noise")+
  #ylab(expression(beta))+
  ylab("Directed exploration")+
  annotate("text",x=3.5,y=c(0.45,0.415),label = label,parse=T))

corTestPretty(subset(data,model=="SG"&SG_best==1)$par,subset(data,model=="SG"&SG_best==1)$beta,method = "kendall")

all_res <- cowplot::plot_grid(cowplot::plot_grid(lc,soc_sd_trial,a,soc_sd,nrow=1,labels ="auto"),
                              cowplot::plot_grid(pxp_ind,pxp_soc,betacomp,expl_replace,eps_soc_rew,
                                                 rel_widths = c(0.5,0.5,1,1,1),nrow = 1,labels = c("e","f","g","h","i")),nrow=2)
ggsave("./plots/all_res_e2.pdf",plot=all_res,width=14,height=7)
