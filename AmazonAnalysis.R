library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(glmnet)
library(rpart)
library(agua)
library(ggmosaic)
library(embed)

train_data <- vroom(
  "GitHub/AmazonEmployeeAccess/amazon-employee-access-challenge/train.csv") %>% 
  mutate(ACTION = factor(ACTION))
test_data <- vroom(
  "GitHub/AmazonEmployeeAccess/amazon-employee-access-challenge/test.csv")
plot1 <- ggplot(train_data, aes(
  x = RESOURCE, 
  y = ACTION)) +
  stat_summary(fun = mean, geom = "col", fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Resource", y = "Access granted") +
  theme_minimal()
plot2 <- ggplot(data = train_data, aes(x = RESOURCE)) +
  geom_bar(fill = "steelblue")+
  theme_minimal()
plot3 <- ggplot(train_data, aes(x = ACTION)) +
  geom_bar(aes(y = after_stat(count) / after_stat(sum(count))), fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "ACTION", y = "Proportion") +
  theme_minimal()
min(train_data$RESOURCE)
max(train_data$RESOURCE)
train_data$RESOURCE <- cut(train_data$RESOURCE, 
                           breaks = c(0, 10000,20000,30000,40000,50000,
                                      60000,70000,80000,90000,Inf),
                           labels = c("0 to 10k","10k to 20k",
                                      "20k to 30k","30k to 40k",
                                      "40k to 50k","50k to 60k","60k to 70k",
                                      "70k to 80k","80k to 90k","Above 90k"),
                           right = FALSE)

median(train_data$RESOURCE)
mean(train_data$RESOURCE)
ultplot<-(plot1 + plot2)/plot3

min(train_data$MGR_ID)
max(train_data$MGR_ID)
mean(train_data$MGR_ID)
median(train_data$MGR_ID)
train_data$MGR_ID <- cut(train_data$MGR_ID, 
                           breaks = c(0, 10000,20000,30000,40000,50000,
                                      60000,70000,80000,90000,Inf),
                           labels = c("0 to 10k","10k to 20k",
                                      "20k to 30k","30k to 40k",
                                      "40k to 50k","50k to 60k","60k to 70k",
                                      "70k to 80k","80k to 90k","Above 90k"),
                           right = FALSE)
ggplot(data = train_data, aes(x = MGR_ID)) +
  geom_bar(fill = "steelblue")+
  theme_minimal()
ggplot(train_data, aes(
  x = MGR_ID, 
  y = ACTION)) +
  stat_summary(fun = mean, geom = "col", fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(x = "Manager ID", y = "Access granted") +
  theme_minimal()
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
  step_dummy(all_factor_predictors()) 

prepped_recipe <- prep(amazon_recipe)
baked_data <- bake(prepped_recipe, new_data=train_data)
ncol(baked_data)

logRegModel <- logistic_reg() %>% 
  set_engine("glm")

logReg_workflow <- workflow() %>% 
  add_recipe(amazon_recipe) %>%
  add_model(logRegModel) %>% 
  fit(data = train_data)

amazon_predictions <- predict(logReg_workflow,
                              new_data=test_data,
                              type="prob") # "class" or "prob"
kaggle_sub <- bind_cols(test_data$id, amazon_predictions[2]) %>% 
  rename(ACTION = .pred_1) %>% 
  rename(id = ...1)

vroom_write(kaggle_sub,
            file="GitHub/AmazonEmployeeAccess/logistical_reg.csv", delim=",")
