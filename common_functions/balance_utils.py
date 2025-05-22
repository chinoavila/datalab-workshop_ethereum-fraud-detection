from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from collections import Counter

def apply_undersampling(X_train, y_train):
    nm = NearMiss()
    return nm.fit_resample(X_train, y_train)

def apply_oversampling(X_train, y_train):
    os = RandomOverSampler()
    return os.fit_resample(X_train, y_train)

def apply_smote_tomek(X_train, y_train):
    st = SMOTETomek()
    return st.fit_resample(X_train, y_train)

def get_balanced_ensemble(random_state=42):
    return BalancedBaggingClassifier(random_state=random_state)
