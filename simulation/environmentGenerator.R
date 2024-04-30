#Environment generator for Social Generalization 
#Charley Wu (Jan 2020)

rm(list=ls())
packages<-c("plyr", 'cowplot', 'rdist', 'ggplot2' , 'viridis', 'jsonlite')
invisible(lapply(packages, require, character.only = TRUE))

#Globally fixed prameters
gridSize <- 11
xstar <- expand.grid(x=1:gridSize, y = 1:gridSize) #input space
lambda <- 2 #length scale

#########################################################################################################################
# Gaussian Process functions
#########################################################################################################################
# kernel function
#l =  length scale (aka lambda)

rbf_D <- function(X1, X2=NULL,l=1){
  if (is.null(X2)){
    D <- pdist(X1)^2
  }else{
    D <- cdist(X1, X2)^2
  }
  Sigma <- exp(-D/(2*l^2))
}

#Gaussian Process function
#lambda is length scale
#eps is error variance
#k allows for selecting other kernels
#full_cov is whether to return a dataframe of the mean and variance of each option, or to return the full covariance matrix (for sampling)
gpr <- function(Xstar,X,Y, lambda, eps = sqrt(.Machine$double.eps), k = rbf_D, full_cov = F ){
  #Compute the covariance between observed inputs
  K <- k(X,X,lambda) #K(x,x') for each pair of observed inputs in X
  KK <- K + diag(eps, nrow(K)) #(K + noise * I)
  KK.inv <- chol2inv(chol(KK)) #Invert matrix using Cholesky decomposition
  Ky <- KK.inv %*% Y # #times y
  if(!full_cov){ #return only the mean and variance vectors
    result <- sapply(Xstar, function(x_i){ 
      #Compute covariance of observed inputs with target space (Xstar)
      Kstar <- k(X, x_i, lambda)
      Kstarstar <- k(x_i,x_i,lambda)  #Covariance of Xstar with itself
      #Compute posterior as a mean vector and a variance vector
      mu <-t(Kstar)  %*% Ky #get mean vector
      var <- Kstarstar - (t(Kstar) %*% KK.inv %*% Kstar) #get covariance
      cbind(mu,var)
    })
    prediction <- as.data.frame(t(result))
    colnames(prediction) <- c('mu', 'var')
    return(prediction) #return it as a data farm
  }else{#return the full covariance matrix
    #Compute covariance of observed inputs with target space (Xstar)
    Kstar <- k(X, Xstar, lambda)
    Kstarstar <- k(Xstar,Xstar,lambda)  #Covariance of Xstar with itself
    #Compute posterior as a mean vector and a variance vector
    mu <-t(Kstar)  %*% Ky #get mean vector
    cov <- Kstarstar - (t(Kstar) %*% KK.inv %*% Kstar) 
    return(list(mu = mu, cov = cov))
  }
}

#Minmax scaling to 0-1
normalize <- function(x){(x-min(x))/(max(x)-min(x))}
#########################################################################################################################
# Simple Environment generation
# Sample non-socially correlated environments from the GP prior
#########################################################################################################################
#Parameters
n_envs <- 10
# compute kernel on pairwise values
Sigma <- rbf_D(xstar,l=lambda)
# sample from multivariate normal with mean zero, sigma = sigma
Z <- MASS::mvrnorm(n_envs,rep(0,dim(Sigma)[1]), Sigma) #Sample a single canonical function 

environmentList <- list()
plot_list = list()
for (i in 1:n_envs){
  z <- normalize(Z[i,]) #scale to 0 and 1
  M <- data.frame(x1 = xstar$x, x2 = xstar$y, payoff = z)
  environmentList[[i]] <- M #add to list
  #plot each env
  plot_list[[i]] <- ggplot(M, aes(x = x1, y = x2, fill = payoff ))+
    geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
    ggtitle(bquote(M[.(i)]^0))
}
#Save plots
payoffplots <- cowplot::plot_grid(plotlist = plot_list, ncol = 8)
ggsave('plots/M0_c09.pdf', payoffplots, width = 12, height = 8)
#Save environments
write_json(environmentList, 'environments/M0_c09_parent.json')


#########################################################################################################################
# Correlated environment generation
# Sample a canonical function from the GP prior, and then use that function as the mean function
# Then sample individual payoff functions from a GP where the mean is defined but the variance is still the prior variance
#########################################################################################################################
#Simulation parameters
genNum <- 5000000
n_players <- 4
n_envs <- 10
correlationThreshold <- .9
tolerance <- .05
childNames = c('A', 'B', 'C', 'D')

# compute kernel on pairwise values
Sigma_social <- rbf_D(xstar,l=lambda)

#M0 generated above are the canonical environments
M <-fromJSON("environments/M0_c09_parent.json", flatten=TRUE) #load from above

childEnvList = list(A=list(), B=list(), C=list(), D=list())
plot_list = list(list(), list(), list(), list())
#Sample functions from the new prior mean is defined by the canonical environment
for (i in 1:n_envs){
  Z_n <- MASS::mvrnorm(genNum,M[[i]][,'payoff'], Sigma_social, ) #generate many candidates
  cors<- sapply(1:genNum, FUN = function(k) cor(M[[i]][,'payoff'], Z_n[k,])) #compute correlations with canonical environment
  #remove environments with correlations lower than threshold with canonical
  Z_n <- Z_n[cors>correlationThreshold,] #-0.15 for .9 because we lose too many envs otherwise
  #Try to find a set of 4 environments, where all envs have above threshold correlations amongst each other
  found <- FALSE 
  while (found==FALSE){
    candidates <- sample(1:nrow(Z_n), size = n_players)
    cors <- c()
    for (k in 2:n_players){
      cors <- c(cors,cor(Z_n[candidates[1],],Z_n[candidates[k],]))
    }
    #Check that all correlations fall within the correlation threshold with A? tolerance
    checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
    if (checked==3){ #now repeat for environment B (with C&D, no need to check AB again --> 2)
      for (k in 3:n_players){
        cors <- c(cors,cor(Z_n[candidates[2],],Z_n[candidates[k],]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      # if (checked==5){
      #   #found <- T
      #   #print("found one")
      # }
      if (checked==5){#finally, check correlation of C&D
        cors <- c(cors,cor(Z_n[candidates[3],],Z_n[candidates[4],]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      if (checked==6){ #all 6 combinations checked --> conditions fulfilled, add to list
        found<-TRUE
        print("found one")
      }
    }
  }
  Z_n <- Z_n[candidates,]
  for (j in 1:n_players){
    Z_j <- normalize(Z_n[j,])
    entry <- data.frame(x1=xstar$x, x2 = xstar$y, payoff=Z_j)
    childEnvList[[childNames[j]]][[i]] <- entry #add to list
    #plot each env
    plot_list[[j]][[i]] <- ggplot(entry, aes(x = x1, y = x2, fill = payoff ))+
      geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
      ggtitle(bquote(.(childNames[[j]])[.(i)]^1))
  }
}

 #Save plots
for (child in childNames){
  i <- match(child, childNames)
  payoffplots <- cowplot::plot_grid(plotlist = plot_list[[i]], ncol = 8)
  ggsave(paste0('plots/', child,'_c09.pdf'), payoffplots, width = 12, height = 8)  
}

#Save environments
for (child in childNames){
  write_json(childEnvList[[child]], paste0('environments/', child,'_c09.json'))
}
 
# #########################################################################################################################
# Correlation between different payoffs in M0
# #########################################################################################################################
# #TODO: Find which pairs of environments are the most uncorrelated
# cors <- c()
# cors_np <- c()
# for (i in 1:40){
#   for (j in 1:40){
#     if (i !=j){
#       #test <- min(cors)
#       cors <- c(cors, cor(c(matrix(M[[i]][,'payoff'], nrow=11)), c(matrix(M[[j]][,'payoff'], nrow=11))))#why was this at nrow 8?
#       cors_np <- c(cors_np, cor(c(matrix(M[[i]][,'payoff'], nrow=11)), c(matrix(M[[j]][,'payoff'], nrow=11)),method='spearman'))
#       #if (test < cor(c(matrix(M[[i]][,'payoff'], nrow=8)), c(matrix(M[[j]][,'payoff'], nrow=8)))) {
#       #  lowest <- c(i,j)
#       #}
#       #if (abs(test) < abs(cor(c(matrix(M[[i]][,'payoff'], nrow=8)), c(matrix(M[[j]][,'payoff'], nrow=8))))) {
#       #  least <- c(i,j)
#       #}
#     }
#   }
# }
# hist(cors)
# hist(cors_np)

# #######################################################################################################################
# Correlation sanity checks
# #######################################################################################################################
M <-fromJSON("environments/M0_c09_parent.json", flatten=TRUE)
envs <- c('A','B','C','D') #, 'D'
envList <- vector('list',length(envs))
for (env in envs){
  envList[[match(env,envs)]] <- fromJSON(paste0("environments/",env,"_c09.json"), flatten=TRUE)
}

#parent child correlations
cor_pc <- c()


for (i in 1:length(envList[[1]])){
  for (j in 1:length(envList)){
    cor_pc <- c(cor_pc, cor(c(M[[i]][,'payoff']), c(envList[[j]][[i]][,'payoff'])))
  }
}
hist(cor_pc)
cor_pc = data.frame(cor_pc)

ggplot(cor_pc,aes(x=cor_pc)) + 
  geom_histogram(bins=75) + 
  theme_classic() + 
  geom_vline(aes(xintercept = correlationThreshold,color="red"))+
  theme(legend.position = 'None')+
  xlab('Correlation between parent and child environments')+
  xlim(0,1)+
  ylab('n')
ggsave("./plots/cpr_pc.pdf")


#child-child-cors
cor_cc <- c()


for (i in 1:length(envList[[1]])){
  for (j in 1:length(envList)){
    for(k in 1:length(envList)){
      if (j==k){
        next
      }
    cor_cc <- c(cor_cc, cor(c(envList[[j]][[i]][,'payoff']), c(envList[[k]][[i]][,'payoff'])))
    }
  }
}

hist(cor_cc)
cor_cc <- data.frame(cor_cc)

ggplot(cor_cc,aes(x=cor_cc)) + 
  geom_histogram(bins=100) + 
  theme_classic() + 
  geom_vline(aes(xintercept = correlationThreshold-tolerance,color="red"))+
  geom_vline(aes(xintercept = correlationThreshold+tolerance,color="red"))+
  theme(legend.position = 'None')+
  xlab('Correlation among child environments')+
  xlim(-0.1,1)+
  ylab('n')
ggsave("./plots/cpr_cc.pdf")

cor_cc_AB <- c()
cor_cc_AC <- c()
cor_cc_AD <- c()
cor_cc_BC <- c()
cor_cc_BD <- c()
cor_cc_CD <- c()

for (i in 1:length(envList[[1]])){
  cor_cc_AB <- c(cor_cc_AB, cor(c(envList[[1]][[i]][,'payoff']), c(envList[[2]][[i]][,'payoff'])))
}
for (i in 1:length(envList[[1]])){
  cor_cc_AC <- c(cor_cc_AC, cor(c(envList[[1]][[i]][,'payoff']), c(envList[[3]][[i]][,'payoff'])))
}
for (i in 1:length(envList[[1]])){
  cor_cc_AD <- c(cor_cc_AD, cor(c(envList[[1]][[i]][,'payoff']), c(envList[[4]][[i]][,'payoff'])))
}
for (i in 1:length(envList[[1]])){
  cor_cc_BC <- c(cor_cc_BC, cor(c(envList[[2]][[i]][,'payoff']), c(envList[[3]][[i]][,'payoff'])))
}
for (i in 1:length(envList[[1]])){
  cor_cc_BD <- c(cor_cc_BD, cor(c(envList[[2]][[i]][,'payoff']), c(envList[[4]][[i]][,'payoff'])))
}
for (i in 1:length(envList[[1]])){
  cor_cc_CD <- c(cor_cc_CD, cor(c(envList[[3]][[i]][,'payoff']), c(envList[[4]][[i]][,'payoff'])))
}
mean(cor_cc_AB)
mean(cor_cc_AC)
mean(cor_cc_AD)
mean(cor_cc_BC)
mean(cor_cc_BD)
mean(cor_cc_CD)



######################################################
#Take 2 - environments with no spatial correlation
######################################################
#Parameters
n_envs <- 40
# sample from multivariate normal with mean zero, sigma = sigma
Z <- lapply(1:n_envs, FUN = function(x) rnorm(gridSize**2)) #Sample a single canonical function 

environmentList <- list()
plot_list = list()
for (i in 1:n_envs){
  z <- normalize(Z[[i]]) #scale to 0 and 1
  M <- data.frame(x1 = xstar$x, x2 = xstar$y, payoff = z)
  environmentList[[i]] <- M #add to list
  #plot each env
  plot_list[[i]] <- ggplot(M, aes(x = x1, y = x2, fill = payoff ))+
    geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
    ggtitle(bquote(M[.(i)]^0))
}
#Save plots
#payoffplots <- cowplot::plot_grid(plotlist = plot_list, ncol = 8)
#ggsave('plots/M0_lambda0.pdf', payoffplots, width = 12, height = 8)
#Save environments
#write_json(environmentList, 'environments/M0_lambda0.json')

#now for the correlated ones
genNum <- 10000
n_players <- 4
n_envs <-40
correlationThreshold <- .6
tolerance <- .05
childNames = c('A', 'B', 'C', 'D')

childEnvList = list(A=list(), B=list(), C=list(), D=list())
plot_list = list(list(), list(), list(), list())

Z <- lapply(1:genNum, FUN = function(x) rnorm(gridSize**2)) #Sample a single canonical function 
socNoise <- rnorm(gridSize**2)
Z_soc <- lapply(1:length(Z), function(x) normalize(Z[[x]]+socNoise))


for (i in 1:n_envs){
  found <- FALSE 
  while (found==FALSE){
    candidates <- sample(1:length(Z_soc), size = n_players)
    cors <- c()
    for (k in 2:n_players){
      cors <- c(cors,cor(Z_soc[[candidates[1]]],Z_soc[[candidates[k]]]))
    }
    #Check that all correlations fall within the correlation threshold with A? tolerance
    checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
    if (checked==3){ #now repeat for environment B (with C&D, no need to check AB again --> 2)
      for (k in 3:n_players){
        cors <- c(cors,cor(Z_soc[[candidates[2]]],Z_soc[[candidates[k]]]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      if (checked==5){#finally, check correlation of C&D
        cors <- c(cors,cor(Z_soc[[candidates[1]]],Z_soc[[candidates[4]]]))
      }
      checked <- sum((cors>=(correlationThreshold-tolerance)) & (cors<=(correlationThreshold+tolerance)))
      if (checked==6){ #all 6 combinations checked --> conditions fulfilled, add to list
        found<-TRUE
        Z_sel <- lapply(1:n_players, FUN=function(x) Z_soc[[candidates[x]]])
        for (j in 1:n_players){
          for(k in 1:n_players){
            if (j==k){
              next
            }
           if (cor(c(Z_sel[[j]]), c(Z_sel[[k]])) < (correlationThreshold-tolerance) | cor(c(Z_sel[[j]]), c(Z_sel[[k]])) > (correlationThreshold+tolerance)){
             found <- FALSE
             break
           }
          }
        }
      }
    }
  }
  Z_sel <- lapply(1:n_players, FUN=function(x) Z_soc[[candidates[x]]])
  for (j in 1:n_players){
    entry <- data.frame(x1=xstar$x, x2 = xstar$y, payoff=Z_sel[[j]])
    childEnvList[[childNames[j]]][[i]] <- entry #add to list
    #plot each env
    plot_list[[j]][[i]] <- ggplot(entry, aes(x = x1, y = x2, fill = payoff ))+
      geom_tile()+ theme_void() + scale_fill_viridis(name='Payoff') + theme(legend.position='none') + 
      ggtitle(bquote(.(childNames[[j]])[.(i)]^1))
  }}

for (child in childNames){
  i <- match(child, childNames)
  payoffplots <- cowplot::plot_grid(plotlist = plot_list[[i]], ncol = 8)
  ggsave(paste0('plots/', child,'_lambda0_soc06.pdf'), payoffplots, width = 12, height = 8)
}

#Save environments
for (child in childNames){
  write_json(childEnvList[[child]], paste0('environments/', child,'_lambda0_soc06.json'))
}

#double-check our correlations

envs <- c('A','B','C', 'D')
envList <- vector('list',length(envs))
for (env in envs){
  envList[[match(env,envs)]] <- fromJSON(paste0("environments/",env,"_lambda0_soc06.json"), flatten=TRUE)
}

#child-child-cors
cor_cc <- c()


for (i in 1:length(envList[[1]])){
  for (j in 1:length(envList)){
    for(k in 1:length(envList)){
      if (j==k){
        next
      }
      cor_cc <- c(cor_cc, cor(c(envList[[j]][[i]][,'payoff']), c(envList[[k]][[i]][,'payoff'])))
    }
  }
}
hist(cor_cc)
cor_cc <- data.frame(cor_cc)

ggplot(cor_cc,aes(x=cor_cc)) + 
  geom_histogram(bins=100) + 
  theme_classic() + 
  geom_vline(aes(xintercept = correlationThreshold-tolerance,color="red"))+
  geom_vline(aes(xintercept = correlationThreshold+tolerance,color="red"))+
  theme(legend.position = 'None')+
  xlab('Correlation among child environments')+
  xlim(0,1)+
  ylab('n')
ggsave("./plots/cpr_cc_lambda0.pdf")


# 
# #########################################################################################################################
# # GRAVEYARD
# # Alternative method: Sample a canonical function from the GP prior, and then fix N points where all participants overlap. 
# # Then sample individual payoff functions from the posterior condition on these N points
# #########################################################################################################################

# #Simulation parameters
# xstar <- seq(1,30)
# lambda <- 8
# numObs <- 5 #number of observations to sample
# 
# # compute squared exponential kernel on pairwise values
# Sigma <- rbf_D(xstar,xstar,l=lambda)
# 
# # sample from multivariate normal with mean zero, sigma = sigma
# Y <- MASS::mvrnorm(1,rep(0,dim(Sigma)[1]), Sigma)
# # plot resulx_its
# pp <- data.frame(y=Y,x=xstar) 
# #Sample a few observations
# obs.x <- sample(xstar, size = numObs, replace = F)
# obs.y <- Y[obs.x]
# obsDF <- data.frame(x = obs.x, y = obs.y)
# #Compute GP predictions
# predictions <- gpr(xstar,obs.x, obs.y, lambda, eps = 0.0001) #just the mean and variance values
# posterior <- gpr(xstar,obs.x, obs.y, lambda, eps = 0.0001, full_cov = T) #also the full covariance
# predictions$x <- xstar #add x column
# predictions$upper <-  predictions$mu+ 2*sqrt(predictions$var) #95% CI
# predictions$lower <-  predictions$mu- 2*sqrt(predictions$var)
# #Sample functions from the posterior
# sampleDF <- data.frame()
# for (i in 1:8){
#   Y_i <- MASS::mvrnorm(1,posterior$mu, posterior$cov)  
#   entry <- data.frame(x = xstar, y = Y_i, sample = i)
#   sampleDF <- rbind(sampleDF, entry)
# }
# sampleDF$sample <- factor(sampleDF$sample)
# 
# ggplot(pp, aes(x = x))+
#   geom_ribbon(data = predictions, aes(x = x, ymax = upper, ymin =lower), alpha = 0.2)+
#   geom_line(data = predictions, aes(x=x, y=mu), linetype = 'dashed')+  
#   geom_line(aes(y=y), size=1.5) +
#   geom_line(data = sampleDF, aes(x = x, y=y, color = sample), alpha = 0.8)+
#   geom_point(data = obsDF, aes(y=y), size = 5, shape = 1)+
#   theme_classic()+
#   theme(legend.position='None')+
#   labs(x = expression(bold(x)), y = 'Payoff')




  