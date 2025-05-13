from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier

def run_model(X_train, X_test, y_train, y_test, balanced=False):
    if balanced:
        model = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg", class_weight="balanced")
    else:
        model = LogisticRegression(C=1.0, penalty='l2', random_state=1, solver="newton-cg")
    model.fit(X_train, y_train)
    return model

def apply_balancing(x_train, y_train, strategy):
    if strategy == 'Under-sampling (NearMiss)':
        balancer = NearMiss()
    elif strategy == 'Over-sampling':
        balancer = RandomOverSampler()
    elif strategy == 'SMOTE-Tomek':
        balancer = SMOTETomek()
    elif strategy == 'Balanced Bagging Classifier':
        return None, None
    else:
        return x_train, y_train
        
    return balancer.fit_resample(x_train, y_train)
