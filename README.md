# Overview:
My stakeholders are the Customer Retention Analytics (CRA) team at a telecoms company 'SyriaTel' where I'm a lead datascientist. The CRA team takes a data-driven approach to enhancing customer satisfaction and decreasing churn (when a customer leaves a business). I'm on a team within a centralized datascience and engineering organization. We support various satellite analytics teams including CRA, who distill our datasets further into business metrics, interactive dashboards, and slide deck presentations. The CRA team supports the greater customer retention org, which then implements changes to our incentives program. 

# Problem and Objective:
We lack the ability to identify customers when they're on the cusp of churning. The objective of this project is to predict likely churners. Our dataset includes 20 variables describing over 3,000 current and churned customers. Achieving this predictive ability will allow us to examine the data on a rolling basis and quickly implement targeted incentivization.

# Methodology:
- We use a predictive logistic regression model rather than multiple linear regression because our target variable is categorical rather than numerical and continuous. The modeling process allows us to iteratively guide how the model 'learns' about our data to improve the accuracy of its predictions. While using a machine learning model isn't 'better' than simpler methods of getting at our problem such as SQL analysis, visualizations or EDA, we're focusing on a predictive modeling approach for now. We justify this prioritization because the sooner our CRA team has a customer list for targeted incentivizing, the better. Customer Retention can offer upgrades to those customers and prevent lost business before it occurs.
- This buys us some time while we take a deeper dive into data cleaning, EDA, and inferential modeling. Those methods will complement our predictive model and help to fine-tune it by shedding light on the actual relationships between the predictors and target, and to what extent a given predictor is impacting churn. This way, our retention efforts will become more specific and strategic. Our aim is to strike a balance between preventing lost business and wasting money on extravagant incentives.
- This project will serve as a solution to identification of customers who are about to churn. Then, we can get into the weeds of discovering the precise correlations and causations of churn. All the while, our predicitve model will be acting as a stop-gap to minimize lost business.

# Data Source:
- [Kaggle: "Churn in Telecom's dataset"](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

# Data Understanding:
- This is a dataset of 3,333 customers from a fictional telecoms company "SyriaTel." The 20 features include geographical information about customers, the time they spend on day vs. evening vs. nighttime calls, whether they have a voice mail or internation plan, and the length of their account number which indicates the relative amount of time they've been a customer. Account number is therefore a great proxy for customer lifetime value.

# Modeling:
- Four models were built: a dummy baseline model, a first simple model (FSM) using logistic regression, the FSM with a target class imbalance adjuster (FSM + SMOTE), and finally, a model produced via GridSearch. - I decided on the below hyperparameters and list of options for each by reading up on those commonly used in the machine learning community for tuning logistic regression models. I settled on the below recommended via [machinelearningmastery.com](https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/)
  

```
search_space=[{'lr__solver':['newton-cg', 'lbfgs', 'liblinear', 'saga', 'sag'],
                'lr__penalty':['none', 'l1', 'l2', 'elasticnet'],
                'lr__C': [100, 10, 1.0, 0.1, 0.01]
               }]
```

- The best `penalty` hyperparameter was found to be `l2` by our grid search. This is not surprising because the model we fed into gridsearch was slightly overfit and could be corrected by increasing regularization/penalty. When a model is overfit, it needs to be more generalized as it has low bias and high variance. An overfit model will not be 'general' enough to effectively predict on unseen data. Luckily, our pipeline included scaling from the get-go. Scaling is necessary for l2 regularization as it is distance-based.
- The `C` value is responsible for defining the strength of regularization/penalty, which here is `l2.`
- The solver chosen is `newton-cg.` The `solver` is a hyperparameter that dictates how we solve the function (loss function) that minimizes error (loss). The point on the loss function where loss is the smallest is the point where, when we take the partial derivative, the slope of that tangent line is equal to 0. `newton-cg` is a way of going about gradient descent that uses a different kind of quadratic function (loss function) to find that sweet spot where the slope of the tangent line is 0.


# Model Evaluation
- Dummy Model ---> FSM: 
  - *Log Loss* decreased from 5.1 ---> 0.3
- FSM ---> SMOTE + FSM:
  - *Log Loss* increased from .3 ---> .51
  - *Accuracy* becomes a useful validation metric because we addressed class imbalance with SMOTE
-  Change from SMOTE + FSM ---> Final Model:
   -  *Log Loss* increased from 0.51 ---> 0.54
   -  *Accuracy* decreased from 0.78 ---> 0.74
-  Final Model Accuracy: Train vs. Test:
   -  Train: 0.79
   -  Test on held-out dataset: 0.74

# Conclusion
This was a good place to start in our endeavor to predict customer churn. We were able to address class imbalance so that Accuracy is now a useful validation measure. We ran a Grid Search to optimize these hyperparameters: Regularization ("penalty"), penalty intensity ("C"), and loss function solver ("solver"). We found that applying Grid Search and using the resulting model with hyperparameters C=0.1, penalty=l2, and solver=newton-cg increased log loss and decreased accuracy compared to the cross-validation results from our SMOTE+FSM model. When examining our final model's performance between train and testing, the accuracy was significantly lower. This result leads us to new questions, in particular about overfitting. Because the degree of overfitting wasn't seen until we tested on the hold-out dataset, we may want to decrease complexity of features to reduce noise, perhaps by grouping "state" by the highest level of LG speed that state has. This might be a better signal and by grouping states into these larger categories we reduce the noise caused by too many features. Another opportunity is to explore other validation metrics such as the F-statistic, Recall, and precision.