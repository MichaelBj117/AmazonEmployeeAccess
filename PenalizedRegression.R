library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(agua)
library(embed)

train_data <- vroom(
  "GitHub/AmazonEmployeeAccess/train.csv") %>% 
  mutate(ACTION = factor(ACTION))
test_data <- vroom(
  "GitHub/AmazonEmployeeAccess/test.csv")



amazon_recipe <- recipe(ACTION ~ ., data=train_data) %>%
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

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

penlog_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

penlogReg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(penlog_mod) 

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- penlogReg_workflow %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- penlogReg_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

amazon_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, amazon_predictions[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/AmazonEmployeeAccess/pen_logistical_reg_batch.csv", delim=",")
