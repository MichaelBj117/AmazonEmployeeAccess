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
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) 

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

forest_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(forest_mod) 

tuning_grid <- grid_regular(mtry(range(c(1,9))),
                            min_n(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- forest_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

amazon_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, amazon_predictions[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/AmazonEmployeeAccess/random_forest.csv", delim=",")
