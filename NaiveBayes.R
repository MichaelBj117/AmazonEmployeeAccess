library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(embed)
library(kknn)
library(discrim)
library(naivebayes)

train_data <- vroom(
  "GitHub/AmazonEmployeeAccess/train.csv") %>% 
  mutate(ACTION = factor(ACTION))
test_data <- vroom(
  "GitHub/AmazonEmployeeAccess/test.csv")



amazon_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  step_mutate_at(all_numeric_predictors() , fn = factor) %>% # turn all numeric features into factors
  step_other(all_factor_predictors(), threshold = .001) %>% # combines categorical values that occur <.1% i
  step_lencode_mixed(all_factor_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors())

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes") 

nb_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nb_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- nb_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

amazon_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, amazon_predictions[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/AmazonEmployeeAccess/NaiveBayes.csv", delim=",")
