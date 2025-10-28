library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(embed)

train_data <- vroom(
  "GitHub/AmazonEmployeeAccess/train.csv") %>% 
  mutate(ACTION = factor(ACTION))
test_data <- vroom(
  "GitHub/AmazonEmployeeAccess/test.csv")

nn_recipe <- recipe(ACTION ~ ., data=train_data) %>%
  update_role(MGR_ID, new_role="id") %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>% 
  step_other(all_nominal_predictors(), threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]

prepped_recipe <- prep(nn_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

nn_model <- mlp(hidden_units = tune(),
                epochs = 10)  %>%
  set_engine("keras") %>% #verbose = 0 prints off less9
  set_mode("classification")

nn_workflow <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 25)),
                            levels=5)

folds <- vfold_cv(train_data, v = 5, repeats=1)

CV_results <- nn_workflow %>%
  tune_grid(resamples=folds,
            grid=nn_tuneGrid,
            metrics=metric_set(roc_auc)) #Or leave metrics NULL

CV_results %>% collect_metrics() %>%
  filter(.metric=="roc_auc") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

final_workflow <- nn_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train_data)

nn_predictions <- predict(final_workflow,
                              new_data=test_data,
                              type="prob") # "class" or "prob"

kaggle_sub <- bind_cols(test_data$id, nn_predictions$.pred_1) %>% 
  rename(ACTION = ...2) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/AmazonEmployeeAccess/Neural.csv", delim=",")
