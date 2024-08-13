Need a method to fill missing values( Imputation, ...)
id                       517788
member_id                517788
emp_title                 34051
emp_length                31300
url                      517788
                          ...
settlement_status        498528
settlement_date          498528
settlement_amount        498528
settlement_percentage    498528
settlement_term          498528
Length: 104, dtype: int64
(517788, 145)

*** numerical - mean, non-numerical - most_frequent


Encoding Categorical values
In general, one-hot encoding (Approach 3) will typically perform best, 
and dropping the categorical columns (Approach 1) typically performs worst, but it varies on a case-by-case basis.
*** used ordinal encoding, easier

Feature Scaling
standard, min-max and robust

***even though standardization happens after filling in missing vlues(imputation) with mean the new mean is the same
   therefore it is OK

Feature Engineering (Selection)
Recursive Feature Elimination, SelectKBest

***RFE
Number of features selected: 5, Selected features: Index(['funded_amnt', 'total_rec_prncp', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt'],
      dtype='object'), Performance score: 0.9995
Number of features selected: 10, Selected features: Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'installment',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt'],
      dtype='object'), Performance score: 0.9994
Number of features selected: 15, Selected features: Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'installment',
       'grade', 'sub_grade', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_int', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt', 'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9992
Number of features selected: 20, Selected features: Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'installment', 'grade', 'sub_grade', 'title', 'total_pymnt',
       'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_amnt', 'last_credit_pull_d', 'avg_cur_bal',
       'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9990
Number of features selected: 25, Selected features: Index(['loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
       'installment', 'grade', 'sub_grade', 'title', 'dti', 'revol_bal',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_d', 'last_pymnt_amnt', 'last_credit_pull_d',
       'total_rev_hi_lim', 'avg_cur_bal', 'tot_hi_cred_lim',
       'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9988



***SelectKBest
Number of features selected: 5, Selected features: Index(['total_pymnt', 'total_rec_prncp', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt'],
      dtype='object'), Performance score: 0.9911
Number of features selected: 10, Selected features: Index(['int_rate', 'grade', 'sub_grade', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_amnt', 'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9947
Number of features selected: 15, Selected features: Index(['term', 'int_rate', 'grade', 'sub_grade', 'verification_status',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_amnt', 'acc_open_past_24mths', 'num_tl_op_past_12m',
       'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9933
Number of features selected: 20, Selected features: Index(['term', 'int_rate', 'grade', 'sub_grade', 'verification_status', 'dti',
       'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',
       'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
       'last_pymnt_amnt', 'last_credit_pull_d', 'acc_open_past_24mths',
       'avg_cur_bal', 'bc_open_to_buy', 'num_tl_op_past_12m',
       'tot_hi_cred_lim', 'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9915
Number of features selected: 25, Selected features: Index(['term', 'int_rate', 'grade', 'sub_grade', 'home_ownership',
       'verification_status', 'dti', 'total_pymnt', 'total_pymnt_inv',
       'total_rec_prncp', 'total_rec_late_fee', 'recoveries',
       'collection_recovery_fee', 'last_pymnt_amnt', 'last_credit_pull_d',
       'tot_cur_bal', 'acc_open_past_24mths', 'avg_cur_bal', 'bc_open_to_buy',
       'mort_acc', 'num_actv_rev_tl', 'num_tl_op_past_12m', 'tot_hi_cred_lim',
       'total_bc_limit', 'debt_settlement_flag'],
      dtype='object'), Performance score: 0.9906

Classifier

SelectKBest without PCA
Best Hyperparameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}
Validation Accuracy: 0.6773447820343461   

SelectKBest with PCA(5 features)
Best Hyperparameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}
Validation Accuracy: 0.935328744582725


RFE with PCA(4 features)
Best Hyperparameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}
Validation Accuracy: 0.997833089990498

RFE without PCA
Best Hyperparameters: {'colsample_bytree': 1.0, 'gamma': 0.1, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1.0}
Validation Accuracy: 0.3032051727734131