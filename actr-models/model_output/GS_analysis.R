library(tidyverse)
rm(list = ls())

# load func
load("./func.RData")

load.mdat <- function(model, subjID) {
  if (model=="model1") {
    m1.gsfiles = list.files(path = ".", pattern = paste(paste("^MODEL1.*", subjID, sep=""), "*_gs.csv$", sep=""), full.names = T)
    res <- read.csv(m1.gsfiles) %>% 
      mutate(CurrentResponse = case_when(Response=="j" ~ 3, Response=="f" ~ 2), RT = as.numeric(RT), Epoch = as.numeric(Epoch),
             ans = replace_na(ans, 'nil')) 
  } else{
    m2.gsfiles = list.files(path = ".", pattern = paste(paste("^MODEL2.*", subjID, sep=""), "*_gs.csv$", sep=""), full.names = T)
    res <- read.csv(m2.gsfiles) %>% 
      mutate(CurrentResponse = case_when(Response=="j" ~ 3, Response=="f" ~ 2), RT = as.numeric(RT),  Epoch = as.numeric(Epoch)) 
  }
  return(res)
}

load.sdat <- function(subjID) {
  gambling.clean <- read_csv("../../bin/gambling_clean_data.csv") %>% 
    filter(HCPID==subjID) %>%
    select(-BlockType) %>% rename(BlockType=BlockTypeCoded) %>%
    filter(!is.na(ResponseSwitch)) %>%
    group_by(HCPID, BlockType, TrialType) %>%
    summarise(PSwitch.subj = mean(ResponseSwitch)) 
}

calc.mLL <- function(subjID, subj.dat, m1.dat, m2.dat){
  # aggregate model data (add sd)
  m1.agg <- clean_previous_future(count_responses(m1.dat)) %>%
    filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
    group_by(ans, bll, lf, Epoch, BlockType, TrialType) %>%
    dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
    ungroup() %>%
    mutate(ParamID = group_indices(., ans, bll, lf)) %>%
    group_by(ParamID, ans, bll, lf, BlockType, TrialType) %>%
    dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
    ungroup() %>%
    mutate(HCPID = subjID) 
  
  m2.agg <- clean_previous_future(count_responses(m2.dat)) %>%
    filter(!is.na(ResponseSwitch) & !is.na(CurrentResponse)) %>%
    group_by(egs, alpha, r, Epoch, BlockType, TrialType) %>%
    dplyr::summarise(PSwitch = mean(ResponseSwitch)) %>%
    ungroup() %>%
    mutate(ParamID = group_indices(., egs, alpha, r)) %>%
    group_by(ParamID, egs, alpha, r, BlockType, TrialType) %>%
    dplyr::summarise(PSwitch.mean = mean(PSwitch), PSwitch.sd = sd(PSwitch)) %>%
    mutate(HCPID = subjID) 
  
  m1.logL <- left_join(m1.agg, subj.dat, by = c("HCPID", "BlockType", "TrialType")) %>%
    mutate(PSwitch.z = (PSwitch.subj-PSwitch.mean)/(PSwitch.sd),
           PSwitch.probz = dnorm(PSwitch.z, 0, 1),
           PSwitch.logprobz = log(PSwitch.probz)) %>% 
    group_by(HCPID, ParamID, ans, bll, lf) %>%
    dplyr::summarise(PSwitch.LL = sum(PSwitch.logprobz)) %>%
    ungroup() %>%
    slice_max(PSwitch.LL, n = 1)
  
  m2.logL <- left_join(m2.agg, subj.dat, by = c("HCPID", "BlockType", "TrialType")) %>%
    mutate(PSwitch.z = (PSwitch.subj-PSwitch.mean)/(PSwitch.sd),
           PSwitch.probz = dnorm(PSwitch.z, 0, 1),
           PSwitch.logprobz = log(PSwitch.probz)) %>% 
    group_by(HCPID, ParamID, egs, alpha, r) %>%
    dplyr::summarise(PSwitch.LL = sum(PSwitch.logprobz)) %>%
    ungroup() %>%
    slice_max(PSwitch.LL, n = 1)
  
  m.logL <- left_join(m1.logL, m2.logL, by = "HCPID", suffix = c(".m1", ".m2")) %>%
    mutate(best_model = ifelse(PSwitch.LL.m1>PSwitch.LL.m2, "m1", "m2"))
  
  return(m.logL)
}

load.HCPID <- function(model){
  if (model=='model1') {
    m1.gsfiles = list.files(path = ".", pattern = "^MODEL1.*_gs.csv$", full.names = F)
    res = c()
    for (f in m1.gsfiles) {
      res <- c(paste(str_split_fixed(f, '_', 3)[2], 'fnca', sep = '_'), res)
    }
  } else {
    m2.gsfiles = list.files(path = ".", pattern = "^MODEL2.*_gs.csv$", full.names = F)
    res = c()
    for (f in m2.gsfiles) {
      res <- c(paste(str_split_fixed(f, '_', 3)[2], 'fnca', sep = '_'), res)
    }
  }
  return(res)
}


runLL <- function() {
  #subjID <- '100307_fnca'
  #subjID <- '100408_fnca'
  m1.HCPIDs = load.HCPID('model1')
  m2.HCPIDs = load.HCPID('model2')
  HCPIDs = m1.HCPIDs
  if (all(m1.HCPIDs==m2.HCPIDs)) {
    done.IDs = read.csv('./MODELLogLikelihood.csv') %>% select(HCPID)
    HCPIDs <- anti_join(data.frame(HCPID = m1.HCPIDs), done.IDs)
  } else{
    missing.IDs <- anti_join(data.frame(HCPID = m1.HCPIDs), data.frame(HCPID = m2.HCPIDs))
    print("Missing some gs files")
    print(missing.IDs)
    done.IDs = read.csv('./MODELLogLikelihood.csv') %>% select(HCPID)
    HCPIDs <- anti_join(data.frame(HCPID = m1.HCPIDs), done.IDs)
    HCPIDs <- anti_join(HCPIDs, missing.IDs)
  }
  
  #HCPIDs <- c('100307_fnca')
  df.LL <- data.frame()
  for (subjID in HCPIDs$HCPID) {
    m1.dat = load.mdat("model1", subjID)
    m2.dat = load.mdat("model2", subjID)
    subj.dat = load.sdat(subjID)
    df.LL <- df.LL %>% 
      rbind(calc.mLL(subjID, subj.dat, m1.dat, m2.dat))
  }
  return (df.LL)
}

ll.append <- function(new.LL) {
  old.LL <- read.csv('./MODELLogLikelihood.csv')
  update.LL <- old.LL %>% rbind(new.LL)
  return(update.LL)
}

df.LL <- runLL()
updated.LL <- ll.append(df.LL) 
write.csv(updated.LL, './MODELLogLikelihood.csv')





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

