library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)




train_data <- vroom(
  "GitHub/AmazonEmployeeAccess/train.csv") %>% 
  mutate(ACTION = factor(ACTION))
test_data <- vroom(
  "GitHub/AmazonEmployeeAccess/test.csv")



amazon_recipe1 <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate(RESOURCE = cut(RESOURCE,
                             breaks = c(0, 10000, 20000, 30000, 40000, 50000,
                                        60000, 70000, 80000, 90000, Inf),
                             right = FALSE)) %>%
  step_mutate(MGR_ID = cut(MGR_ID, 
                           breaks = c(0, 10000,20000,30000,40000,50000,
                                      60000,70000,80000,90000,Inf),
                           right = FALSE)) %>% 
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) 

amazon_recipe2 <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION))

prepped_recipe1 <- prep(amazon_recipe1)
baked_data1 <- bake(prepped_recipe, new_data=train_data)

prepped_recipe2 <- prep(amazon_recipe2)
baked_data2 <- bake(prepped_recipe, new_data=train_data)

penlog_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penlogReg_workflow1 <- workflow() %>% 
  add_recipe(amazon_recipe1) %>%
  add_model(penlog_mod) 

penlogReg_workflow2 <- workflow() %>% 
  add_recipe(amazon_recipe1) %>%
  add_model(penlog_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results1 <- penlogReg_workflow1 %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

CV_results2 <- penlogReg_workflow2 %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune1 <- CV_results1 %>%
  select_best(metric = "roc_auc")

bestTune2 <- CV_results2 %>%
  select_best(metric = "roc_auc")

final_workflow1 <- penlogReg_workflow %>%
  finalize_workflow(bestTune1) %>%
  fit(data=train_data)

final_workflow2 <- penlogReg_workflow %>%
  finalize_workflow(bestTune2) %>%
  fit(data=train_data)

amazon_predictions1 <- predict(final_workflow1,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

amazon_predictions2 <- predict(final_workflow2,
                               new_data=test_data,
                               type="prob") # "class" or "prob"

kaggle_sub1 <- bind_cols(test_data$id, amazon_predictions1[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_sub2 <- bind_cols(test_data$id, amazon_predictions2[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub1, file="./pen_logistical_reg_batch1.csv", delim=",")
vroom_write(kaggle_sub2, file="./pen_logistical_reg_batch2.csv", delim=",")



############################################################################
#linear regression

amazon_recipe3 <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate(RESOURCE = cut(RESOURCE,
                             breaks = c(0, 10000, 20000, 30000, 40000, 50000,
                                        60000, 70000, 80000, 90000, Inf),
                             right = FALSE)) %>%
  step_mutate(MGR_ID = cut(MGR_ID, 
                           breaks = c(0, 10000,20000,30000,40000,50000,
                                      60000,70000,80000,90000,Inf),
                           right = FALSE)) %>% 
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_factor_predictors()) 

amazon_recipe4 <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_factor_predictors()) 

prepped_recipe3 <- prep(amazon_recipe3)
baked_data3 <- bake(prepped_recipe3, new_data=train_data)

prepped_recipe4 <- prep(amazon_recipe4)
baked_data4 <- bake(prepped_recipe4, new_data=train_data)

logRegModel <- logistic_reg() %>% 
  set_engine("glm")

logReg_workflow1 <- workflow() %>% 
  add_recipe(amazon_recipe3) %>%
  add_model(logRegModel) %>% 
  fit(data = train_data)

logReg_workflow2 <- workflow() %>% 
  add_recipe(amazon_recipe4) %>%
  add_model(logRegModel) %>% 
  fit(data = train_data)

amazon_predictions3 <- predict(logReg_workflow1,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

amazon_predictions4 <- predict(logReg_workflow2,
                               new_data=test_data,
                               type="prob") # "class" or "prob"

kaggle_sub3 <- bind_cols(test_data$id, amazon_predictions3[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_sub4 <- bind_cols(test_data$id, amazon_predictions4[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub3, file="./logistical_reg1.csv", delim=",")
vroom_write(kaggle_sub4, file="./logistical_reg2.csv", delim=",")