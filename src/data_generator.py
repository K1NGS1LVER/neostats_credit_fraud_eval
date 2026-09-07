"""
High-fidelity stratified Home Credit Default Risk dataset generator.
Generates 10,000 realistic records replicating authentic Kaggle distributions,
inter-feature correlations, missingness profiles, and 8.07% default rate.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_home_credit_sample(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. IDs
    sk_ids = np.arange(100001, 100001 + n_samples)
    
    # 2. Demographics & Categoricals
    gender_p = [0.65, 0.35]
    genders = np.random.choice(['F', 'M'], size=n_samples, p=gender_p)
    
    contract_p = [0.90, 0.10]
    contract_types = np.random.choice(['Cash loans', 'Revolving loans'], size=n_samples, p=contract_p)
    
    car_p = [0.34, 0.66]
    own_car = np.random.choice(['Y', 'N'], size=n_samples, p=car_p)
    
    realty_p = [0.69, 0.31]
    own_realty = np.random.choice(['Y', 'N'], size=n_samples, p=realty_p)
    
    children_choices = [0, 1, 2, 3, 4]
    children_p = [0.70, 0.20, 0.08, 0.015, 0.005]
    cnt_children = np.random.choice(children_choices, size=n_samples, p=children_p)
    
    income_types = ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student']
    income_p = [0.52, 0.23, 0.18, 0.069, 0.001]
    name_income_type = np.random.choice(income_types, size=n_samples, p=income_p)
    
    education_types = [
        'Secondary / secondary special',
        'Higher education',
        'Incomplete higher',
        'Lower secondary',
        'Academic degree'
    ]
    edu_p = [0.71, 0.24, 0.033, 0.015, 0.002]
    name_edu_type = np.random.choice(education_types, size=n_samples, p=edu_p)
    
    family_types = ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow']
    fam_p = [0.64, 0.15, 0.10, 0.06, 0.05]
    name_fam_type = np.random.choice(family_types, size=n_samples, p=fam_p)
    
    housing_types = ['House / apartment', 'With parents', 'Municipal apartment', 'Rented apartment', 'Office apartment']
    housing_p = [0.88, 0.05, 0.04, 0.02, 0.01]
    name_housing_type = np.random.choice(housing_types, size=n_samples, p=housing_p)
    
    occupation_types = [
        'Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers',
        'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff',
        'Cooking staff', 'Cleaning staff'
    ]
    occ_p = [0.30, 0.18, 0.14, 0.12, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01]
    occupations = np.random.choice(occupation_types, size=n_samples, p=occ_p).astype(object)
    
    # 3. Continuous Variables
    ages = np.random.beta(a=3, b=3, size=n_samples) * (68 - 21) + 21
    days_birth = (-ages * 365.25).astype(int)
    
    days_employed = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if name_income_type[i] == 'Pensioner':
            days_employed[i] = 365243  # Standard Home Credit code for retired
            occupations[i] = np.nan
        else:
            max_work_years = max(1, int(ages[i] - 18))
            work_years = np.random.exponential(scale=5.0)
            work_years = min(work_years, max_work_years)
            days_employed[i] = int(-work_years * 365.25)
            
    for i in range(n_samples):
        if name_income_type[i] != 'Pensioner' and np.random.rand() < 0.20:
            occupations[i] = np.nan
            
    log_income = np.random.normal(loc=11.9, scale=0.55, size=n_samples)
    amt_income_total = np.round(np.exp(log_income) / 1000) * 1000
    for i in range(n_samples):
        if name_edu_type[i] == 'Higher education':
            amt_income_total[i] *= 1.35
        elif name_edu_type[i] == 'Academic degree':
            amt_income_total[i] *= 1.80
        elif name_edu_type[i] == 'Lower secondary':
            amt_income_total[i] *= 0.85
    amt_income_total = np.clip(amt_income_total, 27000, 2000000)
    
    credit_multiplier = np.random.uniform(1.8, 6.0, size=n_samples)
    amt_credit = np.round((amt_income_total * credit_multiplier) / 5000) * 5000
    amt_credit = np.clip(amt_credit, 45000, 3000000)
    
    payment_rate = np.random.normal(loc=0.052, scale=0.012, size=n_samples)
    payment_rate = np.clip(payment_rate, 0.025, 0.10)
    amt_annuity = np.round((amt_credit * payment_rate) / 100) * 100
    
    goods_diff = np.random.uniform(0.85, 1.05, size=n_samples)
    amt_goods_price = np.round((amt_credit * goods_diff) / 5000) * 5000
    
    days_registration = (-np.random.exponential(scale=4000, size=n_samples) - 100).astype(int)
    days_id_publish = (-np.random.uniform(100, 6000, size=n_samples)).astype(int)
    days_last_phone_change = (-np.random.exponential(scale=900, size=n_samples)).astype(int)
    region_rating = np.random.choice([1, 2, 3], size=n_samples, p=[0.12, 0.73, 0.15])
    
    ext_source_2 = np.random.beta(a=3.5, b=3.5, size=n_samples)
    ext_source_3 = np.random.beta(a=3.2, b=3.2, size=n_samples)
    ext_source_1 = np.random.beta(a=3.0, b=3.0, size=n_samples).astype(object)
    
    mask_ext_1 = np.random.rand(n_samples) < 0.56
    ext_source_1[mask_ext_1] = np.nan
    
    mask_ext_3 = np.random.rand(n_samples) < 0.20
    ext_source_3 = ext_source_3.astype(object)
    ext_source_3[mask_ext_3] = np.nan
    
    def_30 = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.88, 0.08, 0.03, 0.01])
    def_60 = np.clip(def_30 - np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]), 0, 3)
    bureau_year = np.random.poisson(lam=1.8, size=n_samples)
    
    z = np.full(n_samples, -2.95)
    es2_val = np.nan_to_num(ext_source_2.astype(float), nan=0.51)
    es3_val = np.nan_to_num(ext_source_3.astype(float), nan=0.51)
    
    z += -2.5 * (es2_val - 0.51)
    z += -2.8 * (es3_val - 0.51)
    
    dti = amt_annuity / amt_income_total
    z += 2.2 * np.maximum(0, dti - 0.20)
    
    pmt_rate = amt_annuity / amt_credit
    z += 12.0 * np.maximum(0, pmt_rate - 0.06)
    
    z += -0.025 * (ages - 43)
    
    for i in range(n_samples):
        if name_income_type[i] != 'Pensioner':
            tenure_years = -days_employed[i] / 365.25
            if tenure_years < 2.0:
                z[i] += 0.45
            elif tenure_years > 8.0:
                z[i] -= 0.35
                
    for i in range(n_samples):
        if name_edu_type[i] in ['Higher education', 'Academic degree']:
            z[i] -= 0.55
        elif name_edu_type[i] == 'Lower secondary':
            z[i] += 0.50
            
    z += 0.35 * def_30
    
    prob_default = 1 / (1 + np.exp(-z))
    target_count = int(np.round(n_samples * 0.0807))
    default_indices = np.argsort(prob_default)[-target_count:]
    targets = np.zeros(n_samples, dtype=int)
    targets[default_indices] = 1
    
    df = pd.DataFrame({
        'SK_ID_CURR': sk_ids,
        'TARGET': targets,
        'NAME_CONTRACT_TYPE': contract_types,
        'CODE_GENDER': genders,
        'FLAG_OWN_CAR': own_car,
        'FLAG_OWN_REALTY': own_realty,
        'CNT_CHILDREN': cnt_children,
        'AMT_INCOME_TOTAL': amt_income_total,
        'AMT_CREDIT': amt_credit,
        'AMT_ANNUITY': amt_annuity,
        'AMT_GOODS_PRICE': amt_goods_price,
        'NAME_INCOME_TYPE': name_income_type,
        'NAME_EDUCATION_TYPE': name_edu_type,
        'NAME_FAMILY_STATUS': name_fam_type,
        'NAME_HOUSING_TYPE': name_housing_type,
        'DAYS_BIRTH': days_birth,
        'DAYS_EMPLOYED': days_employed,
        'DAYS_REGISTRATION': days_registration,
        'DAYS_ID_PUBLISH': days_id_publish,
        'OCCUPATION_TYPE': occupations,
        'REGION_RATING_CLIENT': region_rating,
        'EXT_SOURCE_1': ext_source_1,
        'EXT_SOURCE_2': ext_source_2,
        'EXT_SOURCE_3': ext_source_3,
        'DEF_30_CNT_SOCIAL_CIRCLE': def_30,
        'DEF_60_CNT_SOCIAL_CIRCLE': def_60,
        'DAYS_LAST_PHONE_CHANGE': days_last_phone_change,
        'AMT_REQ_CREDIT_BUREAU_YEAR': bureau_year
    })
    
    return df

if __name__ == '__main__':
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True, parents=True)
    df = generate_home_credit_sample(n_samples=10000, random_state=42)
    csv_path = out_dir / 'sample_application.csv'
    df.to_csv(csv_path, index=False)
    print(f"Dataset generated at {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Default rate: {df['TARGET'].mean():.4%}")
