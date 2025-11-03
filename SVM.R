library(tidyverse)
library(tidymodels)
library(vroom)
library(kernlab)


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
  step_pca(all_predictors(), threshold=0.3) #Threshold is between 0 and 1

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

## SVM models
svmRadial <- svm_rbf(rbf_sigma = 0.177, cost = 0.00316) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmPoly <- svm_poly(degree = 1, cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svmLinear <- svm_linear(cost = 0.0131) %>%
  set_mode("classification") %>%
  set_engine("kernlab")



svmR_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(svmRadial) %>%
  fit(data=train_data)

svmP_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(svmPoly) %>%
  fit(data=train_data)

svmL_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(svmLinear) %>%
  fit(data=train_data)


amazon_predictionsP <- predict(svmP_workflow,
                               new_data=test_data,
                               type="prob") # "class" or "prob"

amazon_predictionsR <- predict(svmR_workflow,
                                new_data=test_data,
                                type="prob")

amazon_predictionsL <- predict(svmL_workflow,
                               new_data=test_data,
                               type="prob")

kaggle_subP <- bind_cols(test_data$id, amazon_predictionsP[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_subR <- bind_cols(test_data$id, amazon_predictionsR[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

kaggle_subL <- bind_cols(test_data$id, amazon_predictionsL[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_subP,
            file="GitHub/AmazonEmployeeAccess/svmPoly.csv", delim=",")

vroom_write(kaggle_subR,
            file=
              "GitHub/AmazonEmployeeAccess/svmRadial.csv", delim=",")

vroom_write(kaggle_subL,
            file="GitHub/AmazonEmployeeAccess/svmLinear.csv", delim=",")
