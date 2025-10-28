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
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_dummy(all_predictors()) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=0.5) #Threshold is between 0 and 1

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

forest_mod <- rand_forest(mtry = tune(),
                          min_n=tune(),
                          trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

penlog_mod <- logistic_reg(mixture=tune(), penalty=tune()) %>% #Type of model
  set_engine("glmnet")

knn_model <- nearest_neighbor(neighbors=tune()) %>% # tune
  set_mode("classification") %>%
  set_engine("kknn")

forest_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(forest_mod) 

penlogReg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(penlog_mod)

knn_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(knn_model)

tuning_gridF <- grid_regular(mtry(range(c(1,9))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

tuning_gridPL <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

tuning_gridK <- grid_regular(neighbors(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_resultsF <- forest_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_gridF,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

CV_resultsPL <- penlogReg_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_gridPL,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

CV_resultsK <- knn_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_gridK,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTuneF <- CV_resultsF %>%
  select_best(metric = "roc_auc")

bestTunePL <- CV_resultsPL %>%
  select_best(metric = "roc_auc")

bestTuneK <- CV_resultsK %>%
  select_best(metric = "roc_auc")

final_workflowF <- forest_workflow %>%
  finalize_workflow(bestTuneF) %>%
  fit(data=train_data)

final_workflowPL <- penlogReg_workflow %>%
  finalize_workflow(bestTunePL) %>%
  fit(data=train_data)

final_workflowK <- knn_workflow %>%
  finalize_workflow(bestTuneK) %>%
  fit(data=train_data)

amazon_predictionsF <- predict(final_workflowF,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

amazon_predictionsPL <- predict(final_workflowPL,
                               new_data=test_data,
                               type="prob")

amazon_predictionsK <- predict(final_workflowK,
                               new_data=test_data,
                               type="prob")

kaggle_subF <- bind_cols(test_data$id, amazon_predictionsF[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_subPL <- bind_cols(test_data$id, amazon_predictionsPL[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_subK <- bind_cols(test_data$id, amazon_predictionsK[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_subF,
            file="GitHub/AmazonEmployeeAccess/random_forestPCA.csv", delim=",")

vroom_write(kaggle_subPL,
            file=
            "GitHub/AmazonEmployeeAccess/pen_logistical_regPCA.csv", delim=",")

vroom_write(kaggle_subK,
            file="GitHub/AmazonEmployeeAccess/KNNPCA.csv", delim=",")
