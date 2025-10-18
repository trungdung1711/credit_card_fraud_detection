from sklearn.model_selection import KFold
import numpy as np

X = np.arange(10)  # 10 samples: [0,1,2,3,4,5,6,7,8,9]

# 5 folds and there is no shuffle
kf = KFold(n_splits=5, shuffle=False)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}")
    print("Val set: ", X[val_idx])
    print("Train set: ", X[train_idx])
    print()
