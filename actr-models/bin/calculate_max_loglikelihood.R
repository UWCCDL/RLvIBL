################### R Script ################### 
# Date: 2021 May
# Author: Cher Yang
# This script calculate Max LogLikelihood
################################################

################### Set up environment ################### 
#.libPaths(c(.libPaths(), "/home/stocco/R/x86_64-pc-linux-gnu-library/3.6"))
library(tidyverse)
library(dplyr)
rm(list = ls())

################### GLOBAL Variables ################### 
#TODO: change path based on running environment
HUMAN_DATA_PATH = "../../behavior/gambling_clean_data.csv"
MODEL_DATA_PATH = "../../actr-models/model_output/model_output_local"
MAXLL_PATH = "../../actr-models/model_output/max_loglikelihood_trial.csv"


################### Load useful functions ################### 
load("./maxLLfunctions.RData")
#setwd("~/Documents/GitProject/RLvIBL/actr-models/model_output")
#setwd('/home/chery/Desktop/RLvIBL/actr-models/model_output_local')


################### Load model data functions ################### 
# This function load model data based on subject id
load.mdat <- function(model, subjID) {
  m.gsfiles = list.files(path = MODEL_DATA_PATH, pattern = paste("^", str_to_upper(model), ".*",subjID, "*_gs.csv$", sep = ""), full.names = T)
  res  <- data.frame()
  for (f in m.gsfiles) {
    res <- res %>% rbind(read.csv(f))
  }
  res <- res %>% 
    mutate(CurrentResponse = case_when(Response=="j" ~ 3, Response=="f" ~ 2), RT = as.numeric(RT),  Epoch = as.numeric(Epoch)) %>%
    filter(TrialType != "Neutral")
  return(res)
}

################### Load subject data functions ################### 
# This function load subject data based on subject id
load.sdat <- function(subjID) {
  gambling.clean <- read_csv(HUMAN_DATA_PATH) %>% 
    filter(HCPID==subjID) %>%
    select(-BlockType) %>% 
    dplyr::rename(BlockType=BlockTypeCoded) %>%
    filter(!is.na(ResponseSwitch)) %>%
    group_by(HCPID, BlockType, TrialType) %>%
    dplyr::summarise(PSwitch.subj = mean(ResponseSwitch)) %>%
    filter(TrialType != "Neutral")
  return(gambling.clean)
}

################### Calculate aggregated stats functions ################### 
# This function calculate aggregated data
# input "model" determines which model to calculate
calculate.model.agg <- function(model, m.dat) {
  if (model=='model1') {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(ans, bll, lf, Epoch, BlockType, TrialType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., ans, bll, lf)) %>%
      group_by(ParamID, ans, bll, lf, BlockType, TrialType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      ungroup() %>%
      mutate(HCPID = subjID)
  } else {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(egs, alpha, r, Epoch, BlockType, TrialType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., egs, alpha, r)) %>%
      group_by(ParamID, egs, alpha, r, BlockType, TrialType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      mutate(HCPID = subjID) 
  }
  return(res)
}

# This function calculate aggregated data for block type
# input "model" determines which model to calculate
calculate.model.block.agg <- function(model, m.dat) {
  if (model=='model1') {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(ans, bll, lf, Epoch, BlockType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., ans, bll, lf)) %>%
      group_by(ParamID, ans, bll, lf, BlockType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      ungroup() %>%
      mutate(HCPID = subjID)
  } else {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(egs, alpha, r, Epoch, BlockType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., egs, alpha, r)) %>%
      group_by(ParamID, egs, alpha, r, BlockType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      mutate(HCPID = subjID) 
  }
  return(res)
}

# This function calculate aggregated data for block type
# input "model" determines which model to calculate
calculate.model.trial.agg <- function(model, m.dat) {
  if (model=='model1') {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(ans, bll, lf, Epoch, TrialType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., ans, bll, lf)) %>%
      group_by(ParamID, ans, bll, lf, TrialType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      ungroup() %>%
      mutate(HCPID = subjID)
  } else {
    res <- clean_previous_future(count_responses(m.dat)) %>%
      filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
      group_by(egs, alpha, r, Epoch, TrialType) %>%
      dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
      ungroup() %>%
      mutate(ParamID = group_indices(., egs, alpha, r)) %>%
      group_by(ParamID, egs, alpha, r, TrialType) %>%
      dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
      mutate(HCPID = subjID) 
  }
  return(res)
}

################### Calculate maxLL functions ################### 
# This function calculate maxLL stats for each individual subject
# input "subjID" determines which subject to calculate
calculate.maxLL <- function(subjID, subj.dat, m1.dat, m2.dat){
  ### both block type and trial type
  #m1.agg <- calculate.model.agg('model1', m1.dat)
  #m2.agg <- calculate.model.agg('model2', m2.dat)
  
  ### only trial aggregate
  aggregate_variables = c("TrialType")
  m1.agg <- calculate.model.trial.agg('model1', m1.dat)
  m2.agg <- calculate.model.trial.agg('model2', m2.dat)
  
  # check param 
  if ((max(m1.agg$ParamID) < 143) | (max(m2.agg$ParamID) < 110)) {
    stop(paste(subjID, 'm1 param:', max(m1.agg$ParamID), 'm2 param:', max(m2.agg$ParamID)))
  } 
  
  
  
  m1.logL <- left_join(m1.agg, subj.dat, by = c("HCPID", aggregate_variables)) %>%
    #filter(TrialType!='Neutral') %>%
    mutate(PSwitch.z = (PSwitch.subj-PSwitch.mean)/max(PSwitch.sd, 1e-10),
           PSwitch.probz = dnorm(PSwitch.z, 0, 1),
           PSwitch.logprobz = log(PSwitch.probz)) %>% 
    group_by(HCPID, ParamID, ans, bll, lf) %>%
    dplyr::summarise(PSwitch.LL = sum(PSwitch.logprobz)) %>%
    ungroup() %>%
    slice_max(PSwitch.LL, n = 1) %>%
    # add model simulation data
    inner_join(m1.agg) %>% #filter(TrialType!='Neutral'), on='ParamID') %>% 
    pivot_wider(id_cols = c('HCPID', 'ParamID', 'ans', 'bll', 'lf', 'PSwitch.LL'), 
                names_from = aggregate_variables, 
                values_from = c('PSwitch.mean', 'PSwitch.sd'))
  
  
  m2.logL <- left_join(m2.agg, subj.dat, by = c("HCPID", aggregate_variables)) %>%
    #filter(TrialType!='Neutral') %>%
    mutate(PSwitch.z = (PSwitch.subj-PSwitch.mean)/max(PSwitch.sd, 1e-10),
           PSwitch.probz = dnorm(PSwitch.z, 0, 1),
           PSwitch.logprobz = log(PSwitch.probz)) %>% 
    group_by(HCPID, ParamID, egs, alpha, r) %>%
    dplyr::summarise(PSwitch.LL = sum(PSwitch.logprobz)) %>%
    ungroup() %>%
    slice_max(PSwitch.LL, n = 1) %>%
    # add model simulation data
    inner_join(m1.agg) %>% #filter(TrialType!='Neutral'), on='ParamID') %>% 
    pivot_wider(id_cols = c('HCPID', 'ParamID', 'ans', 'bll', 'lf', 'PSwitch.LL'), 
                names_from = aggregate_variables, 
                values_from = c('PSwitch.mean', 'PSwitch.sd'))
  
  m.logL <- left_join(m1.logL, m2.logL, by = "HCPID", suffix = c(".m1", ".m2")) %>%
    # add subj data
    left_join(subj.dat %>% #filter(TrialType!='Neutral') %>%
                pivot_wider(id_cols = c('HCPID'),  
                            values_fn = mean,  # only block effect
                            names_from = aggregate_variables, 
                            values_from = c('PSwitch.subj')) %>%
                rename_at(vars(-c('HCPID')), ~paste0(., '.subj')),
              by = 'HCPID') %>%
    mutate(best_model = ifelse(PSwitch.LL.m1>PSwitch.LL.m2, "m1", "m2"),
           maxLL_diff = abs(PSwitch.LL.m1 - PSwitch.LL.m2))
  return(m.logL)
}

################### Load subjID functions ################### 
# This function find common subject ID for both human data, and model data
# input "model" determines which model to calculate
load.HCPID <- function(model){
  if (model=='model1') {
    m1.gsfiles = list.files(path = MODEL_DATA_PATH, pattern = "^MODEL1.*_gs.csv$", full.names = F)
    res = c()
    for (f in m1.gsfiles) {
      res <- c(paste(str_split_fixed(f, '_', 3)[2], 'fnca', sep = '_'), res)
    }
  } else {
    m2.gsfiles = list.files(path = MODEL_DATA_PATH, pattern = "^MODEL2.*_gs.csv$", full.names = F)
    res = c()
    for (f in m2.gsfiles) {
      res <- c(paste(str_split_fixed(f, '_', 3)[2], 'fnca', sep = '_'), res)
    }
  }
  return(res)
}


################### ################### ################### 
#
#   Start calculating ...
#
################### ################### ################### 

runLL <- function() {
  #subjID <- '100307_fnca'
  #subjID <- '100408_fnca'
  m1.HCPIDs = data.frame('HCPID' = load.HCPID('model1')) %>% distinct()
  m2.HCPIDs = data.frame('HCPID' = load.HCPID('model2')) %>% distinct()
  
  if (all(m1.HCPIDs == m2.HCPIDs)) {
    if (file.exists(MAXLL_PATH)) {
      done.IDs <- read.csv(MAXLL_PATH) %>% select(HCPID)
    } else {
      done.IDs <- data.frame(HCPID = '')
    }
    HCPIDs <- anti_join(m1.HCPIDs, done.IDs)
  } else{
    missing.IDs <- anti_join(data.frame(HCPID = m1.HCPIDs), data.frame(HCPID = m2.HCPIDs))
    print("Missing some gs files")
    print(missing.IDs)
  }
  
  #HCPIDs <- c('127630_fnca')
  df.LL <- data.frame()
  for (subjID in HCPIDs$HCPID) {
    m1.dat = try(load.mdat("model1", subjID), silent = T)
    m2.dat = try(load.mdat("model2", subjID), silent = T)
    subj.dat = load.sdat(subjID)
    res.mLL = try(calculate.maxLL(subjID, subj.dat, m1.dat, m2.dat))
    if (inherits(m1.dat, 'try-error') | inherits(m2.dat, 'try-error') | inherits(res.mLL, 'try-error') ) {
      print('skip')
      next
    }
    df.LL <- df.LL %>% 
      rbind(res.mLL)
  }
  return (df.LL)
}

ll.append <- function(new.LL) {
  if (file.exists(MAXLL_PATH)) {
    old.LL <- read.csv(MAXLL_PATH, row.names = 1)
  } else {
    old.LL <- data.frame()
  }
  
  update.LL <- old.LL %>% rbind(new.LL)
  return(update.LL)
}


################### ################### ################### 
#
#   Saving Max LL data
#
################### ################### ################### 
df.LL <- runLL()
update.LL <- ll.append(df.LL) 
write.csv(update.LL, MAXLL_PATH)





# # visualize m1.agg
# ggplot(m1.agg, aes(x = TrialType, y = PSwitch.mean, col = ParamID)) +
#   facet_grid( ~ BlockType) +
#   geom_line(aes(group = ParamID), alpha=.5) + 
#   geom_point(alpha=.5, size=3) +
#   ylim(0,1) +
#   theme_pander()
# 
# ggplot(m2.agg, aes(x = TrialType, y = PSwitch.mean, col = ParamID)) +
#   facet_grid( ~ BlockType) +
#   geom_line(aes(group = ParamID), alpha=.5) + 
#   geom_point(alpha=.5, size=3) +
#   ylim(0,1) +
#   theme_pander()

