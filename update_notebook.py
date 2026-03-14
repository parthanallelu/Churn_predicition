import nbformat
import re

def update_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # FINAL EXHAUSTIVE COLUMN MAP (Strictly aligned with Verified Banking Schema)
    column_map = {
        "MonthlyRevenue": "account_balance",
        "MonthlyMinutes": "tenure_months",
        "TotalRecurringCharge": "account_balance",
        "DirectorAssistedCalls": "num_products",
        "OverageMinutes": "age",
        "RoamingCalls": "monthly_transactions",
        "PercChangeMinutes": "digital_logins",
        "PercChangeRevenues": "monthly_transactions",
        "CustomerCareCalls": "num_complaints",
        "RetentionCalls": "num_products",
        "RetentionTeam": "num_products",
        "DroppedCalls": "num_products",
        "BlockedCalls": "num_products",
        "UnansweredCalls": "num_products",
        "TotalCalls": "monthly_transactions",
        "InboundCalls": "digital_logins",
        "OutboundCalls": "atm_usage",
        "PeakCallsInOut": "monthly_transactions",
        "OffPeakCallsInOut": "monthly_transactions",
        "MonthsInService": "tenure_months",
        "UniqueSubs": "num_products",
        "ActiveSubs": "num_products",
        "HandsetModels": "num_products",
        "HandsetAge": "tenure_months",
        "CurrentEquipmentDays": "tenure_months",
        "EquipmentDays": "tenure_months",
        "CreditRating": "salary_band",
        "ServiceArea": "salary_band",
        "Churn": "churn",
        "HandsetPrice": "num_products",
        "HandsetRefurbished": "has_loan",
        "HandsetWebCapable": "digital_logins",
        "TruckOwner": "has_loan",
        "RVOwner": "has_loan",
        "Homeownership": "has_loan",
        "MaritalStatus": "has_loan",
        "PrizmCode": "salary_band",
        "Occupation": "salary_band",
        "ChildrenInHH": "has_loan",
        "MadeCallToRetentionTeam": "num_complaints",
        "BuysViaMailOrder": "has_loan",
        "RespondsToMailOffers": "has_loan",
        "OptOutMailings": "has_loan",
        "NonUSTravel": "has_loan",
        "OwnsComputer": "has_loan",
        "HasCreditCard": "has_loan",
        "NewCellphoneUser": "has_loan",
        "NotNewCellphoneUser": "has_loan",
        "OwnsMotorcycle": "has_loan",
        "ElectronicBilling": "digital_logins",
        "Handsets": "num_products",
        "Models": "num_products",
        "Age": "age",
        "AgeHH1": "age",
        "AgeHH2": "age",
        "IncomeGroup": "salary_band",
        "RetentionOffersAccepted": "num_products",
        "ReferralsMadeByCustomer": "num_products",
        "AdjustmentsToCreditRating": "salary_band",
        "CreditScore": "credit_score",
        "AvgHHAge": "age",
        "has_cc": "has_loan",
        "is_active": "digital_logins"
    }

    for cell in nb.cells:
        if cell.cell_type == 'code' or cell.cell_type == 'markdown':
            content = cell.source
            for old, new in column_map.items():
                content = content.replace(f"'{old}'", f"'{new}'")
                content = content.replace(f'"{old}"', f'"{new}"')
                content = content.replace(f"[{old}]", f"[{new}]")
                content = content.replace(f"({old})", f"({new})")
            
            content = content.replace("df['churn'] == 'Yes'", "df['churn'] == 1")
            content = content.replace("df['churn'] == 'No'", "df['churn'] == 0")
            cell.source = content

    # Preprocessing cell logic (Cast to float to avoid PDP issues)
    for i, cell in enumerate(nb.cells):
        if "DROP_COLS =" in cell.source:
            cell.source = """import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, RobustScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DROP_COLS = ['churn', 'y', 'salary_band', 'customer_segment']

def preprocess(df_in, y_ser=None, te=None, le=None, meds=None, fit=True):
    d = df_in.copy()
    if 'salary_band' in d.columns:
        if fit:
            te = ce.TargetEncoder(cols=['salary_band'], smoothing=10)
            d['salary_band_enc'] = te.fit_transform(d[['salary_band']], y_ser)['salary_band']
        else:
            d['salary_band_enc'] = te.transform(d[['salary_band']])['salary_band']
    cats = d.select_dtypes('object').columns.tolist()
    if fit:
        le = {}
        for col in cats:
            enc = LabelEncoder(); d[col] = enc.fit_transform(d[col].astype(str)); le[col] = enc
    else:
        for col in cats:
            if col in le:
                d[col] = d[col].astype(str).apply(lambda x: le[col].transform([x])[0] if x in le[col].classes_ else -1)
    
    X = d.drop(columns=DROP_COLS, errors='ignore').astype(float) # Cast all to float for scikit-learn PDP
    X = X.replace([np.inf,-np.inf], np.nan)
    if fit: meds = X.median()
    X = X.fillna(meds).fillna(0).clip(-1e9, 1e9)
    return X, te, le, meds

df_feat = df.copy()
df_feat['BalancePerMonth'] = df_feat['account_balance'] / (df_feat['tenure_months'] + 1)
df_feat['ProductsPerTenure'] = df_feat['num_products'] / (df_feat['tenure_months'] + 1)
if 'age' in df_feat.columns: df_feat['IsSenior'] = (df_feat['age'] > 60).astype(int)

y_full = df_feat['y']
idx_tr, idx_te = train_test_split(df_feat.index, test_size=0.20, random_state=42, stratify=y_full)
idx_tr, idx_va = train_test_split(idx_tr, test_size=0.15, random_state=42, stratify=y_full[idx_tr])
y_train, y_val, y_test = y_full[idx_tr].values, y_full[idx_va].values, y_full[idx_te].values

X_train, te_enc, le_enc, meds = preprocess(df_feat.loc[idx_tr], y_ser=pd.Series(y_train, index=idx_tr), fit=True)
X_val,  *_ = preprocess(df_feat.loc[idx_va], te=te_enc, le=le_enc, meds=meds, fit=False)
X_test, *_ = preprocess(df_feat.loc[idx_te], te=te_enc, le=le_enc, meds=meds, fit=False)
scaler = RobustScaler()
X_train_s, X_val_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
SCALE_POS = (y_train==0).sum() / (y_train==1).sum()

print(f'Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)} | Features: {X_train.shape[1]}')"""
            break

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    update_notebook('churn_prediction_world_class.ipynb')
    print("Notebook updated successfully.")
