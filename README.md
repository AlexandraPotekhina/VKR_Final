# New Composite Materials' Qualities 

The project is for the Data Science course at Bauman Moscow State Technical University.

The data consists of two datasets with numerical features, such as etc.

After performing EDA on raw data, I experimented with diferent preprocessing methods, e.g removed outliers, log-transformed and normalized data. 

The following models were build: Linear Regression, Random Forest Regressor, Gradient Boosting Regressor to predict The Modulus of Elasticity at Tension. For Matrix-Filler Ratio prediction I build a dense/fully connected neural network with tensorflow.keras. Both scikit-learn models and keras model were used in hyperparameter tuning with Grid Search and Random Search to find best parameters. 

The metrics used were: RMSE, MAE, MAPE and MSE (keras model). Models' performance were evaluated on train and test sets.

The best-performing keras model's weights were saved and used in Flask application. The app asks to enter 11 (can be 12) parameters of composite material, and predicts the Matrix-Filler Ratio.

No linear relationships between features were identified. Models did not manage to capture & describe relationships in data. Further experiments are required, e.g. feature engineering new features, creating synthetic data, trying unsupervised learning etc. 


