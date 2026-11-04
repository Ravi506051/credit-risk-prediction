import streamlit as st, pandas as pd, numpy as np, plotly.express as px
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
st.title("Credit Risk Prediction Lab"); st.warning("Synthetic educational demo only; not a real credit score or lending decision.")
X,y=make_classification(n_samples=4000,n_features=6,n_informative=4,weights=[.82,.18],random_state=24); cols=["income_stability","debt_ratio","payment_history","utilization","loan_to_income","account_age"]
a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); m=LogisticRegression(max_iter=1000,class_weight="balanced").fit(a,c); p=m.predict_proba(b)[:,1]
st.metric("ROC-AUC",f"{roc_auc_score(d,p):.3f}")
st.plotly_chart(px.bar(pd.DataFrame({"feature":cols,"coefficient":m.coef_[0]}),x="coefficient",y="feature",orientation="h"),use_container_width=True)
vals={k:st.slider(k,-3.,3.,0.) for k in cols}; st.metric("Illustrative probability",f"{m.predict_proba(pd.DataFrame([vals]))[0,1]:.1%}")
