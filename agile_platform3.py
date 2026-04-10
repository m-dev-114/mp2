"""
╔══════════════════════════════════════════════════════════════╗
║  AGILE INTELLIGENCE PLATFORM  (Dashboard — Website 2)        ║
║  Forest green + cream UI · Fetches from API Server           ║
║  Run: streamlit run agile_platform.py                        ║
║  Requires API running at http://localhost:8000               ║
╚══════════════════════════════════════════════════════════════╝
"""
import streamlit as st
import pandas as pd
import numpy as np
import requests, re, os, pathlib, warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                               RandomForestClassifier, VotingClassifier,
                               AdaBoostClassifier, IsolationForest)
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report, mean_squared_error,
                              r2_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Page config FIRST ─────────────────────────────────────────────────────
st.set_page_config(page_title="Agile Dashboard",
                   layout="wide", page_icon="◆",
                   initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════════════════════
# DESIGN — Forest green + cream, editorial-luxe
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Reset ── */
*,*::before,*::after{box-sizing:border-box;}
html,body{height:100%;}

/* ── Base ── */
.stApp{
  background:#f5f0e8;
  color:#1a2420;
  font-family:'DM Sans','Trebuchet MS',sans-serif;
}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0!important;max-width:100%!important;}
[data-testid="stAppViewContainer"]{background:#f5f0e8;}
[data-testid="stHeader"]{display:none;}

/* ── Nav bar ── */
.topnav{
  position:sticky;top:0;z-index:999;
  background:#1a3c2e;
  border-bottom:2px solid #2d6147;
  padding:0 2.5rem;height:56px;
  display:flex;align-items:center;justify-content:space-between;
}
.nav-logo{
  font-size:1rem;font-weight:700;letter-spacing:0.06em;
  color:#f5f0e8;display:flex;align-items:center;gap:10px;
  font-family:'DM Serif Display','Georgia',serif;
}
.nav-gem{width:9px;height:9px;background:#e8b84b;
  border-radius:2px;transform:rotate(45deg);}
.nav-pills{display:flex;gap:8px;align-items:center;}
.npill{font-size:0.6rem;letter-spacing:0.1em;text-transform:uppercase;
  padding:3px 11px;border-radius:2px;font-weight:600;font-family:monospace;}
.ng{background:#4a7c59;color:#f5f0e8;}
.na{background:transparent;color:#e8b84b;border:1px solid #e8b84b40;}
.nb{background:transparent;color:#a8c8a0;border:1px solid #a8c8a040;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.25}}
.bdot{display:inline-block;width:6px;height:6px;border-radius:50%;
  background:#e8b84b;margin-right:5px;animation:blink 2s infinite;}

/* ── Section header ── */
.shdr{
  font-size:0.6rem;font-weight:700;letter-spacing:0.18em;
  text-transform:uppercase;color:#4a7c59;
  display:flex;align-items:center;gap:10px;margin:1.4rem 0 0.8rem;
}
.shdr::after{content:'';flex:1;height:1px;background:#ddd5c4;}

/* ── Page title ── */
.ptitle{font-family:'DM Serif Display','Georgia',serif;
  font-size:1.6rem;color:#1a3c2e;letter-spacing:-0.02em;font-weight:400;}
.psub{font-size:0.75rem;color:#8a8070;margin-top:4px;letter-spacing:0.03em;}

/* ── Cards ── */
.card{background:#ede7d9;border:1px solid #ddd5c4;
  border-radius:3px;padding:1.1rem 1.3rem;margin-bottom:0.6rem;
  position:relative;overflow:hidden;}
.card::before{content:'';position:absolute;top:0;left:0;right:0;
  height:1px;background:linear-gradient(90deg,transparent,#4a7c5930,transparent);}
.card.red   {border-left:3px solid #c0392b;background:#f9f0ee;}
.card.yellow{border-left:3px solid #d97706;background:#faf5ec;}
.card.green {border-left:3px solid #2d6147;background:#edf5ef;}
.card.blue  {border-left:3px solid #1a3c2e;background:#edf2f0;}
.card.forest{border-left:3px solid #4a7c59;background:#edf5ef;}
.ct{font-size:0.82rem;font-weight:700;color:#1a3c2e;margin-bottom:2px;}
.cd{font-size:0.75rem;color:#8a8070;line-height:1.5;}

/* ── Metric cards ── */
.mc{background:#ede7d9;border:1px solid #ddd5c4;border-radius:3px;
  padding:1rem 1.2rem;position:relative;}
.mc::after{content:'';position:absolute;bottom:0;left:0;right:0;
  height:2px;background:var(--ac,#4a7c59);opacity:0.5;}
.mv{font-size:1.7rem;font-weight:900;color:#1a3c2e;line-height:1;
  font-family:'DM Serif Display','Georgia',serif;}
.ml{font-size:0.6rem;color:#8a8070;text-transform:uppercase;
  letter-spacing:0.1em;margin-top:4px;}
.md{font-size:0.7rem;margin-top:2px;}

/* ── Progress ── */
.pw{background:#ddd5c4;border-radius:2px;height:4px;margin-top:3px;}
.pf{height:4px;border-radius:2px;transition:width 0.6s;}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:4px;padding:2px 9px;
  border-radius:2px;font-size:0.6rem;font-weight:700;
  letter-spacing:0.07em;text-transform:uppercase;font-family:monospace;}
.bg-g{background:#2d614718;color:#2d6147;border:1px solid #2d614730;}
.bg-r{background:#c0392b18;color:#c0392b;border:1px solid #c0392b30;}
.bg-y{background:#d9770618;color:#c4973a;border:1px solid #d9770630;}
.bg-f{background:#4a7c5918;color:#4a7c59;border:1px solid #4a7c5930;}
.bg-a{background:#e8b84b18;color:#c4973a;border:1px solid #e8b84b30;}

/* ── Prediction card ── */
.pred{background:#ede7d9;border:1px solid #ddd5c4;border-radius:3px;
  padding:1.8rem;text-align:center;margin-top:1rem;}
.pred-lbl{font-size:0.58rem;text-transform:uppercase;letter-spacing:0.14em;
  color:#8a8070;margin-bottom:8px;font-family:monospace;}
.pred-val{font-size:1.5rem;font-weight:900;font-family:'DM Serif Display','Georgia',serif;}
.pred-conf{font-size:0.72rem;color:#8a8070;margin-top:4px;}

/* ── API status box ── */
.apibox{background:#1a3c2e;border-radius:3px;padding:1.2rem 1.5rem;
  color:#a8c8a0;font-family:monospace;font-size:0.75rem;margin-bottom:1rem;}
.apibox a{color:#e8b84b;text-decoration:none;}
.apibox a:hover{text-decoration:underline;}

/* ── Assignee card ── */
.acard{background:#ede7d9;border:1px solid #ddd5c4;border-radius:3px;
  padding:0.9rem 0.7rem;text-align:center;}
.av{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:0.88rem;font-weight:900;margin:0 auto 6px;
  font-family:'DM Serif Display','Georgia',serif;}
.an{font-size:0.8rem;font-weight:700;color:#1a3c2e;}
.as{font-size:1.3rem;font-weight:900;line-height:1;
  font-family:'DM Serif Display','Georgia',serif;}
.ad{font-size:0.63rem;color:#8a8070;margin-top:2px;}

/* ── Spark pill ── */
.spill{display:inline-flex;align-items:center;gap:3px;background:#4a7c5918;
  color:#4a7c59;border:1px solid #4a7c5930;border-radius:20px;
  padding:1px 9px;font-size:0.6rem;font-weight:700;
  letter-spacing:0.06em;margin:2px;font-family:monospace;}

/* ── Streamlit overrides ── */
[data-testid="stMetricValue"]{
  font-family:'DM Serif Display','Georgia',serif!important;
  font-size:1.5rem!important;color:#1a3c2e!important;}
[data-testid="stMetricLabel"]{
  font-size:0.6rem!important;color:#8a8070!important;
  text-transform:uppercase;letter-spacing:0.1em;}
[data-testid="stMetricDelta"]{font-size:0.7rem!important;}

div[data-testid="stTabs"] [role="tablist"]{
  background:#ede7d9;border-bottom:2px solid #ddd5c4;padding:0 2rem;}
div[data-testid="stTabs"] button[role="tab"]{
  font-family:monospace!important;font-size:0.62rem!important;
  letter-spacing:0.08em!important;text-transform:uppercase!important;
  color:#8a8070!important;padding:0.7rem 1.1rem!important;
  border:none!important;border-bottom:2px solid transparent!important;
  border-radius:0!important;background:transparent!important;}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{
  color:#1a3c2e!important;border-bottom-color:#4a7c59!important;}

.stButton>button{
  background:#1a3c2e!important;color:#f5f0e8!important;
  border:none!important;border-radius:2px!important;
  font-family:monospace!important;font-size:0.68rem!important;
  letter-spacing:0.08em!important;font-weight:700!important;
  padding:0.5rem 1.3rem!important;text-transform:uppercase!important;}
.stButton>button:hover{background:#2d6147!important;}

.stTextInput>div>div>input,.stNumberInput>div>div>input{
  background:#ede7d9!important;border:1px solid #ddd5c4!important;
  border-radius:2px!important;color:#1a2420!important;
  font-family:monospace!important;font-size:0.78rem!important;}
.stTextInput>div>div>input:focus,.stNumberInput>div>div>input:focus{
  border-color:#4a7c59!important;box-shadow:0 0 0 2px #4a7c5920!important;}

.stSelectbox>div>div{
  background:#ede7d9!important;border:1px solid #ddd5c4!important;
  color:#1a2420!important;border-radius:2px!important;}

[data-testid="stExpander"]{
  background:#ede7d9!important;border:1px solid #ddd5c4!important;
  border-radius:3px!important;}
[data-testid="stExpander"] summary{
  font-size:0.65rem!important;text-transform:uppercase!important;
  letter-spacing:0.08em!important;color:#8a8070!important;}

.stAlert{border-radius:3px!important;font-size:0.78rem!important;}
.stDataFrame{border-radius:3px!important;}
.stSlider>div>div>div{background:#4a7c59!important;}
.stRadio>div{gap:0.5rem!important;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SPARK SETUP
# ══════════════════════════════════════════════════════════════════════════
def _find_java():
    if os.environ.get("JAVA_HOME"): return os.environ["JAVA_HOME"]
    for b in [r"C:\Program Files\Eclipse Adoptium",r"C:\Program Files\Java",
               "/usr/lib/jvm","/usr/local/opt","/Library/Java/JavaVirtualMachines"]:
        p=pathlib.Path(b)
        if p.exists():
            for c in sorted(p.iterdir(),reverse=True):
                jb=c/"bin"/("java.exe" if os.name=="nt" else "java")
                if not jb.exists(): jb=c/"Contents"/"Home"/"bin"/"java"
                if jb.exists(): return str(jb.parent.parent)
    return None

_jh=_find_java()
if _jh:
    os.environ["JAVA_HOME"]=_jh
    os.environ["PATH"]=str(pathlib.Path(_jh)/"bin")+os.pathsep+os.environ.get("PATH","")
os.environ.setdefault("PYSPARK_PYTHON","python3")
os.environ.setdefault("SPARK_LOCAL_IP","127.0.0.1")
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    SPARK_OK=True
except ImportError:
    SPARK_OK=False

@st.cache_resource
def get_spark():
    if not SPARK_OK: return None
    try:
        sp=(SparkSession.builder.appName("AgileAI").master("local[*]")
            .config("spark.driver.memory","2g").config("spark.sql.shuffle.partitions","4")
            .config("spark.ui.enabled","false").config("spark.driver.host","127.0.0.1")
            .config("spark.driver.bindAddress","127.0.0.1").getOrCreate())
        sp.sparkContext.setLogLevel("ERROR"); return sp
    except: return None

SF=["Velocity_Efficiency","Completion_Gap","Blocker_Severity",
    "Scope_Pressure","Sprint_Momentum","Recovery_Index","Workload_Stress"]

def eng_features(pdf, spark=None):
    if spark:
        try:
            sdf=spark.createDataFrame(pdf)
            sdf=sdf.withColumn("Velocity_Efficiency",F.when(F.col("Planned_Story_Points_Sprint")>0,
                F.col("Historical_Velocity")/F.col("Planned_Story_Points_Sprint")).otherwise(1.0))
            sdf=sdf.withColumn("Completion_Gap",
                F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
            sdf=sdf.withColumn("Blocker_Severity",F.col("Blocked_Stories")*F.when(
                F.col("Days_Remaining_Sprint")>0,F.lit(1.0)/F.col("Days_Remaining_Sprint")).otherwise(1.0))
            sdf=sdf.withColumn("Scope_Pressure",F.when(F.col("Planned_Story_Points_Sprint")>0,
                F.col("Scope_Change")/F.col("Planned_Story_Points_Sprint")).otherwise(0.0))
            sdf=sdf.withColumn("Sprint_Momentum",F.when(F.col("Historical_Velocity")>0,
                F.col("Completed_Story_Points")/F.col("Historical_Velocity")).otherwise(0.0))
            sdf=sdf.withColumn("Recovery_Index",F.when(
                (F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points")>0)&
                (F.col("Days_Remaining_Sprint")>0),
                (F.col("Historical_Velocity")*F.col("Days_Remaining_Sprint")/10)/
                (F.col("Planned_Story_Points_Sprint")-F.col("Completed_Story_Points"))
                ).otherwise(1.0))
            sdf=sdf.withColumn("Workload_Stress",
                (F.col("Current_Workload_Percent")/100)*F.col("Consecutive_Overloads"))
            return sdf.toPandas().fillna(0)
        except: pass
    df=pdf.copy()
    df["Velocity_Efficiency"]=(df["Historical_Velocity"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(0,3)
    df["Completion_Gap"]=df["Planned_Story_Points_Sprint"]-df["Completed_Story_Points"]
    df["Blocker_Severity"]=df["Blocked_Stories"]*(1/df["Days_Remaining_Sprint"].replace(0,1).abs())
    df["Scope_Pressure"]=(df["Scope_Change"]/df["Planned_Story_Points_Sprint"].replace(0,1)).clip(-1,2)
    df["Sprint_Momentum"]=(df["Completed_Story_Points"]/df["Historical_Velocity"].replace(0,1)).clip(0,2)
    df["Recovery_Index"]=((df["Historical_Velocity"]*df["Days_Remaining_Sprint"]/10)/
        (df["Planned_Story_Points_Sprint"]-df["Completed_Story_Points"]).replace(0,.001)).clip(0,5)
    df["Workload_Stress"]=(df["Current_Workload_Percent"]/100)*df.get(
        "Consecutive_Overloads",pd.Series(0,index=df.index))
    return df.fillna(0)

# ══════════════════════════════════════════════════════════════════════════
# API CLIENT — fetches from the data website
# ══════════════════════════════════════════════════════════════════════════
class AgileAPIClient:
    def __init__(self,base):
        self.base=base.rstrip("/")
        self.s=requests.Session()
        self.s.headers["Accept"]="application/json"
    def _get(self,path,p=None):
        try:
            r=self.s.get(f"{self.base}{path}",params=p,timeout=12)
            r.raise_for_status(); return r.json(),None
        except requests.exceptions.ConnectionError:
            return None,f"Cannot reach {self.base} — is the API server running?"
        except requests.exceptions.HTTPError as e:
            return None,f"HTTP {e.response.status_code}"
        except Exception as e:
            return None,str(e)
    def _post(self,path,body):
        try:
            r=self.s.post(f"{self.base}{path}",json=body,timeout=12)
            r.raise_for_status(); return r.json(),None
        except requests.exceptions.HTTPError as e:
            return None,f"HTTP {e.response.status_code}: {e.response.text[:150]}"
        except Exception as e: return None,str(e)

    def health(self):
        d,e=self._get("/api/health"); return (True,d.get("version","ok")) if not e else (False,e)
    def fetch_dataset(self,limit=500):
        d,e=self._get("/api/dataset/ml",{"limit":limit})
        if e: return None,e
        return pd.DataFrame(d["records"]),None
    def fetch_sprints(self):
        d,e=self._get("/api/sprints"); return (d.get("sprints",[]),None) if not e else ([],e)
    def fetch_team(self):
        d,e=self._get("/api/team"); return (d.get("team",[]),None) if not e else ([],e)
    def fetch_summary(self): return self._get("/api/analytics/summary")
    def fetch_velocity(self): return self._get("/api/analytics/velocity")
    def fetch_burnout(self): return self._get("/api/analytics/burnout")
    def search(self,q="",assignee=None,priority=None,status=None):
        p={"q":q,"limit":100}
        if assignee: p["assignee"]=assignee
        if priority: p["priority"]=priority
        if status:   p["status"]=status
        d,e=self._get("/api/search",p); return (d.get("results",[]),None) if not e else ([],e)
    def refresh(self): return self._get("/api/refresh")
    def create_issue(self,summary,itype,priority,assignee,sp,labels):
        return self._post("/api/issues",{"summary":summary,"issue_type":itype,
            "priority":priority,"assignee":assignee,"story_points":sp,"labels":labels})
    def add_comment(self,iid,text): return self._post(f"/api/issues/{iid}/comment",{"text":text})
    def transition(self,iid,status): return self._post(f"/api/issues/{iid}/transition",{"status":status})

# ══════════════════════════════════════════════════════════════════════════
# ML ENGINE
# ══════════════════════════════════════════════════════════════════════════
MIN_ROWS=50  # lower threshold so CSV with ~175 rows still trains

@st.cache_data(show_spinner=False)
def tfidf_feat(texts_t,n=15):
    texts=list(texts_t)
    clean=[re.sub(r"[^a-z0-9 ]"," ",str(t).lower()) for t in texts]
    try:
        tv=TfidfVectorizer(max_features=n,ngram_range=(1,2),min_df=2,sublinear_tf=True)
        X=tv.fit_transform(clean).toarray()
        return X,tv.get_feature_names_out().tolist(),tv
    except: return np.zeros((len(texts),n)),[],None

def finetune_lr(Xtr,ytr,Xte,yte):
    best,bm,rows=0,None,[]
    for cfg in [
        dict(C=0.1,solver="lbfgs",    class_weight="balanced",max_iter=300),
        dict(C=1.0,solver="lbfgs",    class_weight="balanced",max_iter=500),
        dict(C=5.0,solver="saga",     class_weight="balanced",max_iter=500),
        dict(C=1.0,solver="liblinear",class_weight="balanced",max_iter=300),
        dict(C=0.5,solver="lbfgs",    class_weight=None,      max_iter=300),
    ]:
        try:
            m=LogisticRegression(**cfg,random_state=42); m.fit(Xtr,ytr)
            yp=m.predict(Xte); acc=accuracy_score(yte,yp); f1=f1_score(yte,yp,average="weighted")
            rows.append({**{k:v for k,v in cfg.items()},"acc":acc,"f1":f1})
            if acc>best: best,bm=acc,m
        except: pass
    return bm,best,rows

@st.cache_data(show_spinner="◆ Training models...")
def train_models(df):
    R={}
    # Sprint — Fine-tuned LR + TF-IDF
    try:
        base=[c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points",
              "Percent_Done","Days_Remaining_Sprint","Historical_Velocity",
              "Blocked_Stories","Scope_Change","Story_Points_Issue",
              "High_Priority_Tasks_Resource"] if c in df.columns]
        txX,txF,tv=None,[],None
        if "Summary" in df.columns and len(df)>=MIN_ROWS:
            txX,txF,tv=tfidf_feat(tuple(df["Summary"].astype(str)),15)
        Xn=df[base].fillna(0).values
        X=np.hstack([Xn,txX]) if txX is not None and len(txF)>0 else Xn
        y=df["Success_Label"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            strat=y if y.nunique()>1 and y.value_counts().min()>=3 else None
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=strat)
            bm,bacc,rows=finetune_lr(Xtr,ytr,Xte,yte); yp=bm.predict(Xte)
            try:    auc=float(roc_auc_score(yte,bm.predict_proba(Xte)[:,1]))
            except: auc=0.0
            R["sprint"]=dict(model=bm,scaler=sc,features=base+(txF if txF else []),
                tfidf=tv,base=base,textf=txF if txF else [],algo="Fine-tuned LR + TF-IDF",
                acc=float(bacc),f1=float(f1_score(yte,yp,average="weighted")),
                auc=auc,report=classification_report(yte,yp),
                cm=confusion_matrix(yte,yp).tolist(),tune=rows)
    except: pass

    # Workload — Naive Bayes
    try:
        f=[c for c in ["Planned_Story_Points_Resource","Current_Assigned_SP","Historical_Avg_SP",
           "Remaining_Days_Resource","High_Priority_Tasks_Resource","Current_Workload_Percent"]
           if c in df.columns]
        X,y=df[f].fillna(0),df["Expected_Overload"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            strat2=y if y.nunique()>1 and y.value_counts().min()>=3 else None
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=strat2)
            m=GaussianNB(); m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["workload"]=dict(model=m,scaler=sc,features=f,algo="Naive Bayes",
                acc=float(accuracy_score(yte,yp)),f1=float(f1_score(yte,yp,average="weighted")),
                report=classification_report(yte,yp),cm=confusion_matrix(yte,yp).tolist())
    except: pass

    # TTR — Ridge
    try:
        Xb=pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex=[c for c in ["Original_Estimate_Hours","Story_Points_Issue"] if c in df.columns]
        X=pd.concat([Xb,df[ex]],axis=1).fillna(0); y=df["Resolution_Time_Hours"]
        if len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
            m=Ridge(alpha=2.0); m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["ttr"]=dict(model=m,scaler=sc,features=X.columns.tolist(),X3=X,
                algo="Ridge Regression",r2=float(r2_score(yte,yp)),
                mse=float(mean_squared_error(yte,yp)))
    except: pass

    # Burnout — Decision Tree
    try:
        f=[c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
           "High_Priority_Tasks_Burnout","Consecutive_Overloads",
           "Current_Workload_Percent"] if c in df.columns]
        X,y=df[f].fillna(0),df["Risk_Flag"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            strat4=y if y.nunique()>1 and y.value_counts().min()>=3 else None
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42,stratify=strat4)
            m=DecisionTreeClassifier(max_depth=4,class_weight="balanced",
                random_state=42,min_samples_leaf=15)
            m.fit(Xtr,ytr); yp=m.predict(Xte)
            R["burnout"]=dict(model=m,scaler=sc,features=f,algo="Decision Tree",
                acc=float(accuracy_score(yte,yp)),f1=float(f1_score(yte,yp,average="weighted")),
                report=classification_report(yte,yp),cm=confusion_matrix(yte,yp).tolist())
    except: pass

    # Allocation — KNN + TF-IDF
    try:
        d2=df.copy()
        les=LabelEncoder(); lel=LabelEncoder()
        d2["Summary_enc"]=les.fit_transform(d2["Summary"].astype(str))
        d2["Labels_enc"] =lel.fit_transform(d2["Labels"].astype(str))
        Xnum=d2[["Summary_enc","Labels_enc","Original_Estimate_Resource","Story_Points_Resource"]].fillna(0)
        if "Summary" in d2.columns and len(d2)>=MIN_ROWS:
            try:
                tva=TfidfVectorizer(max_features=20,ngram_range=(1,1),min_df=2)
                Xt=tva.fit_transform(d2["Summary"].astype(str)).toarray()
                Xa=np.hstack([Xnum.values,Xt])
            except: Xa=Xnum.values
        else: Xa=Xnum.values
        y=d2["Assignee_Resource"]
        if len(d2)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(Xa)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
            k=max(3,min(7,len(Xtr)//10))
            m=KNeighborsClassifier(n_neighbors=k,weights="distance",n_jobs=-1)
            m.fit(Xtr,ytr)
            R["alloc"]=dict(model=m,scaler=sc,features=Xnum.columns.tolist(),
                le_s=les,le_l=lel,algo="KNN + TF-IDF",
                acc=float(accuracy_score(yte,m.predict(Xte))),
                f1=float(f1_score(yte,m.predict(Xte),average="weighted")))
    except: pass
    return R

@st.cache_data(show_spinner=False)
def train_spark_ml(df):
    R={}; sf=[c for c in SF if c in df.columns]
    # Sprint ensemble
    try:
        f=[c for c in ["Planned_Story_Points_Sprint","Completed_Story_Points","Percent_Done",
           "Days_Remaining_Sprint","Historical_Velocity","Blocked_Stories","Scope_Change"]+sf
           if c in df.columns]
        X,y=df[f].fillna(0),df["Success_Label"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.25,random_state=42)
            lr=LogisticRegression(max_iter=300,class_weight="balanced",random_state=42)
            gbt=GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf=RandomForestClassifier(n_estimators=60,random_state=42,n_jobs=-1)
            ada=AdaBoostClassifier(n_estimators=60,random_state=42)
            ens=VotingClassifier([("lr",lr),("gbt",gbt),("rf",rf),("ada",ada)],voting="soft")
            ens.fit(Xtr,ytr)
            ind={}
            for nm,clf in [("LR",lr),("GBT",gbt),("RF",rf),("AdaBoost",ada)]:
                clf.fit(Xtr,ytr); ind[nm]=float(accuracy_score(yte,clf.predict(Xte)))
            gbt.fit(Xtr,ytr)
            imp=pd.Series(gbt.feature_importances_,index=f).sort_values(ascending=False)
            R["sprint"]=dict(acc=float(accuracy_score(yte,ens.predict(Xte))),
                ind=ind,feat=f,sf=sf,imp=imp.to_dict())
    except: pass
    # TTR GBT
    try:
        Xb=pd.get_dummies(df[["Issue_Type","Priority"]],drop_first=False)
        ex=[c for c in ["Original_Estimate_Hours","Story_Points_Issue"]+sf if c in df.columns]
        X=pd.concat([Xb,df[ex]],axis=1).fillna(0); y=df["Resolution_Time_Hours"]
        sc=StandardScaler(); Xs=sc.fit_transform(X)
        Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
        gbr=GradientBoostingRegressor(n_estimators=60,random_state=42); gbr.fit(Xtr,ytr)
        lr3=LinearRegression(); lr3.fit(Xtr,ytr)
        R["ttr"]=dict(gb_r2=float(r2_score(yte,gbr.predict(Xte))),
            lr_r2=float(r2_score(yte,lr3.predict(Xte))),
            gb_mse=float(mean_squared_error(yte,gbr.predict(Xte))),feat=X.columns.tolist(),sf=sf)
    except: pass
    # Burnout GBT+RF
    try:
        f=[c for c in ["Total_SP_This_Sprint","Historical_Avg_SP_Burnout",
           "High_Priority_Tasks_Burnout","Consecutive_Overloads"]+sf if c in df.columns]
        X,y=df[f].fillna(0),df["Risk_Flag"]
        if y.nunique()>1 and len(df)>=MIN_ROWS:
            sc=StandardScaler(); Xs=sc.fit_transform(X)
            Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.25,random_state=42)
            gbt4=GradientBoostingClassifier(n_estimators=60,random_state=42)
            rf4=RandomForestClassifier(n_estimators=60,class_weight="balanced",random_state=42,n_jobs=-1)
            ens4=VotingClassifier([("gbt",gbt4),("rf",rf4)],voting="soft")
            ens4.fit(Xtr,ytr); yp=ens4.predict(Xte)
            R["burnout"]=dict(acc=float(accuracy_score(yte,yp)),
                report=classification_report(yte,yp),feat=f,sf=sf)
    except: pass
    # K-Means++
    try:
        if "Assignee" in df.columns:
            agg={}
            if "Current_Workload_Percent" in df.columns: agg["Workload"]=("Current_Workload_Percent","mean")
            if "Risk_Flag" in df.columns:               agg["Burnout"]=("Risk_Flag","mean")
            if "Success_Label" in df.columns:           agg["SprintRisk"]=("Success_Label",lambda x:(x==0).mean())
            if "Consecutive_Overloads" in df.columns:   agg["ConsecOL"]=("Consecutive_Overloads","mean")
            if len(agg)>=2:
                adf=df.groupby("Assignee").agg(**agg).fillna(0)
                if len(adf)>=2:
                    from sklearn.metrics import silhouette_score
                    Xc=StandardScaler().fit_transform(adf)
                    max_k=min(6,len(adf)-1); kr=list(range(2,max_k+1))
                    iner,sils=[],[]
                    for k in kr:
                        km=KMeans(n_clusters=k,init="k-means++",random_state=42,n_init=10)
                        km.fit(Xc); iner.append(km.inertia_)
                        try: sils.append(float(silhouette_score(Xc,km.labels_)))
                        except: sils.append(0.0)
                    bk=kr[int(np.argmax(sils))] if sils else 2
                    km_f=KMeans(n_clusters=bk,init="k-means++",random_state=42,n_init=15)
                    adf["Cluster"]=km_f.fit_predict(Xc)
                    nc=adf.select_dtypes(include="number").columns.tolist()
                    R["cluster"]=dict(agg=adf.to_dict(),num_cols=nc,best_k=bk,
                        k_range=kr,inertias=iner,sils=sils,
                        sil=sils[kr.index(bk)] if bk in kr else 0.0)
    except: pass
    # Anomaly
    try:
        af=[c for c in ["Current_Workload_Percent","Blocked_Stories","Consecutive_Overloads",
                          "Completion_Gap","Blocker_Severity","Workload_Stress"] if c in df.columns]
        if len(af)>=2:
            iso=IsolationForest(contamination=0.05,random_state=42,n_estimators=100)
            sc=iso.fit_predict(df[af].fillna(0)); cf=iso.score_samples(df[af].fillna(0))
            R["anomaly"]=dict(count=int((sc==-1).sum()),feats=af,
                scores=sc.tolist(),confs=cf.tolist())
    except: pass
    return R

@st.cache_data(show_spinner=False)
def bert_classify(texts_t):
    texts=list(texts_t)
    clean=[re.sub(r"[^a-z0-9 ]"," ",str(t).lower()) for t in texts]
    kw={"Bug":["fix","error","bug","crash","null","fail","broken","exception"],
        "Feature":["add","new","feature","implement","create","support","enhance"],
        "Tech-Debt":["refactor","cleanup","optimize","migrate","debt","restructure"],
        "Performance":["slow","performance","latency","speed","timeout","memory"],
        "Security":["auth","security","permission","token","encrypt","vulnerability"],
        "Regression":["regression","revert","rollback","restore","broken","undo"]}
    preds,terms=[],{}
    tv=TfidfVectorizer(max_features=150,ngram_range=(1,2),min_df=1)
    try: Xt=tv.fit_transform(clean); vocab=tv.get_feature_names_out()
    except: Xt,vocab=None,[]
    for text in clean:
        sc={cat:sum(1 for w in ws if w in text) for cat,ws in kw.items()}
        best=max(sc,key=sc.get); preds.append(best if sc[best]>0 else "Other")
    if Xt is not None:
        for cat in set(preds):
            idx=[i for i,p in enumerate(preds) if p==cat]
            if idx:
                ms=Xt[idx].mean(axis=0).A1; ti=ms.argsort()[-6:][::-1]
                terms[cat]=[vocab[i] for i in ti]
    return preds,terms

# ══════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════
for k,v in [("api_url","http://localhost:8000"),("api_ok",False),
            ("client",None),("df",None),("summary",None),
            ("sprints",[]),("team",[]),("velocity",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════
# NAV BAR
# ══════════════════════════════════════════════════════════════════════════
_spark=get_spark()
spark_lbl="SPARK ACTIVE" if (SPARK_OK and _spark) else "PANDAS MODE"
api_cls="ng" if st.session_state.api_ok else "na"
api_lbl=f"API ✓" if st.session_state.api_ok else "API OFFLINE"

st.markdown(f"""
<div class="topnav">
  <div class="nav-logo"><div class="nav-gem"></div>Agile Dashboard</div>
  <div class="nav-pills">
    <span class="npill nb">⚡ {spark_lbl}</span>
    <span class="npill {api_cls}"><span class="bdot"></span>{api_lbl}</span>
    <span class="npill na">◆ v3.0</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
T=st.tabs(["◆ CONNECT","◈ OVERVIEW","① SPRINT","② WORKLOAD",
           "③ TIME TO RESOLVE","④ BURNOUT","⑤ ALLOCATION",
           "◉ BERT TEXT","⚡ SPARK ML","◎ EVAL","✦ WRITE","⊞ LIVE DATA"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 0 — CONNECT
# ══════════════════════════════════════════════════════════════════════════
with T[0]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Connect to Data Website</div>
      <div class='psub'>Fetch live agile data from the Agile Intelligence API Server</div>
    </div>""",unsafe_allow_html=True)

    st.markdown("""<div style='padding:1rem 2.5rem;'>""",unsafe_allow_html=True)

    # API info box
    st.markdown(f"""<div class='apibox'>
      <b style='color:#e8b84b;'>◆ START THE API SERVER FIRST</b><br><br>
      Terminal 1 (API Server):<br>
      <span style='color:#a8c8a0;'>uvicorn agile_api_server:app --reload --port 8000</span><br><br>
      Then visit the data website: <a href='http://localhost:8000' target='_blank'>http://localhost:8000</a><br>
      Swagger docs: <a href='http://localhost:8000/docs' target='_blank'>http://localhost:8000/docs</a>
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns([3,1])
    with c1:
        api_url_in=st.text_input("API Server URL",st.session_state.api_url,
            placeholder="http://localhost:8000",key="url_in")
    with c2:
        limit_in=st.selectbox("Records",["200","300","500","1000"],index=2,key="lim_in")

    bc1,bc2,bc3=st.columns(3)
    with bc1:
        if st.button("◆ CONNECT & FETCH",key="conn",use_container_width=True):
            client=AgileAPIClient(api_url_in)
            ok,msg=client.health()
            if not ok:
                st.error(f"✗ {msg}")
                st.session_state.api_ok=False
            else:
                with st.spinner("Fetching data from API..."):
                    df_r,e=client.fetch_dataset(int(limit_in))
                    if e: st.error(f"✗ Dataset: {e}")
                    else:
                        smry,_=client.fetch_summary()
                        sprs,_=client.fetch_sprints()
                        team,_=client.fetch_team()
                        vel,_=client.fetch_velocity()
                        st.session_state.update({"api_url":api_url_in,"api_ok":True,
                            "client":client,"df":df_r,"summary":smry,
                            "sprints":sprs or [],"team":team or [],"velocity":vel})
                        st.success(f"✓ Connected to {api_url_in} — {len(df_r):,} records loaded")
                        st.rerun()
    with bc2:
        if st.button("↺ REFRESH DATA",key="ref",use_container_width=True):
            if st.session_state.client:
                st.session_state.client.refresh()
                df_r,e=st.session_state.client.fetch_dataset(int(limit_in))
                if not e:
                    smry,_=st.session_state.client.fetch_summary()
                    st.session_state.df=df_r; st.session_state.summary=smry
                    st.success(f"✓ Refreshed — {len(df_r):,} records"); st.rerun()
    with bc3:
        if st.session_state.api_ok:
            st.markdown(f"""<div style='padding:0.5rem;text-align:center;'>
              <span class='badge bg-g'>● CONNECTED</span></div>""",unsafe_allow_html=True)

    if st.session_state.api_ok:
        h=st.session_state.summary or {}
        st.markdown("<div class='shdr'>API STATUS</div>",unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Records",f"{len(st.session_state.df):,}" if st.session_state.df is not None else "—")
        c2.metric("Sprints",str(len(st.session_state.sprints)))
        c3.metric("Team Members",str(len(st.session_state.team)))
        c4.metric("Health Score",f"{h.get('health_score',0):.0f}/100")
        # CSV download link
        base_url=st.session_state.api_url
        st.markdown(f"""<div class='card forest' style='padding:0.8rem 1.2rem;margin-top:0.5rem;'>
          <span style='font-size:0.75rem;color:#4a7c59;'>
            ◆ Download dataset as CSV: &nbsp;
            <a href='{base_url}/api/dataset/csv' target='_blank'
               style='color:#2d6147;font-family:monospace;font-weight:700;'>
               {base_url}/api/dataset/csv ↗</a>
            &nbsp;— or upload the downloaded file in the CSV tab below.
          </span>
        </div>""",unsafe_allow_html=True)

        st.markdown("<div class='shdr'>QUICK LINKS</div>",unsafe_allow_html=True)
        base=st.session_state.api_url
        eps=[("/","Data Website"),("/docs","Swagger"),("/api/dataset/ml","ML Dataset"),
             ("/api/analytics/summary","Analytics"),("/api/sprints","Sprints"),("/api/team","Team")]
        ec=st.columns(len(eps))
        for i,(path,lbl) in enumerate(eps):
            with ec[i]:
                st.markdown(f"""<div class='card' style='text-align:center;padding:0.7rem;'>
                  <div style='font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;
                    color:#8a8070;margin-bottom:4px;'>{lbl}</div>
                  <a href='{base}{path}' target='_blank'
                     style='color:#2d6147;font-family:monospace;font-size:0.68rem;
                            text-decoration:none;font-weight:700;'>{path}</a>
                </div>""",unsafe_allow_html=True)

# ── CSV Upload section (below connect tab, always visible) ───────────────
st.markdown("<hr style='border:none;border-top:1px solid #e0d8c8;margin:0;'>",unsafe_allow_html=True)
with st.expander("📂  Upload CSV instead (from API → /api/dataset/csv or your own file)", expanded=st.session_state.df is None):
    st.markdown("""<div class='card forest' style='margin-bottom:0.8rem;'>
      <div class='ct'>How to get the CSV</div>
      <div class='cd'>
        <b>Option 1</b> — Start API server then visit:
        <code style='color:#2d6147;'>http://localhost:8000/api/dataset/csv</code><br>
        <b>Option 2</b> — Upload <code>agile_small_200.csv</code> sample file.<br>
        <b>Option 3</b> — Upload any agile CSV with the required ML columns.
      </div>
    </div>""",unsafe_allow_html=True)
    csv_file = st.file_uploader("Choose a CSV file", type="csv",
                                 label_visibility="collapsed", key="csv_upload")
    if csv_file is not None:
        try:
            df_csv = pd.read_csv(csv_file).fillna(0)
            st.session_state.df      = df_csv
            st.session_state.api_ok  = False
            st.session_state.sprints = []
            st.session_state.team    = []
            st.session_state.summary = None
            required = ["Success_Label","Expected_Overload","Risk_Flag","Resolution_Time_Hours"]
            missing  = [c for c in required if c not in df_csv.columns]
            st.success(f"✓ {len(df_csv):,} rows · {len(df_csv.columns)} columns loaded from {csv_file.name}")
            if missing:
                st.warning(f"⚠ Missing columns: {', '.join(missing)} — some models unavailable.")
            else:
                st.success("✓ All required ML columns present.")
            st.dataframe(df_csv.head(4), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Guard
if st.session_state.df is None:
    for i in range(1,12):
        with T[i]:
            st.markdown("""<div style='text-align:center;padding:6rem 0;'>
              <div style='font-family:"DM Serif Display","Georgia",serif;font-size:3rem;
                color:#ddd5c4;margin-bottom:1rem;'>◆</div>
              <div style='font-size:0.85rem;color:#8a8070;'>
                Connect the API above or upload a CSV to begin.</div>
            </div>""",unsafe_allow_html=True)
    st.stop()

# Prep
df=st.session_state.df.copy().fillna(0)
for col,thresh in [("Success_Label",0.5),("Expected_Overload",0.5),("Risk_Flag",0.3)]:
    if col in df.columns:
        if df[col].dtype==object: df[col]=df[col].map({"No":0,"Yes":1}).fillna(0).astype(int)
        else: df[col]=(df[col]>thresh).astype(int)
df=eng_features(df,_spark)
sf_ok=[c for c in SF if c in df.columns]
M=train_models(df); SM=train_spark_ml(df)
tp,tt=[],{}
if "Summary" in df.columns: tp,tt=bert_classify(tuple(df["Summary"].astype(str)))

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with T[1]:
    h=st.session_state.summary or {}
    pills="".join([f"<span class='spill'>⚡{f}</span>" for f in sf_ok[:4]])
    st.markdown(f"""<div style='padding:2rem 2.5rem 0;'>
      <div style='margin-bottom:8px;'>
        <span class='badge bg-g'>◆ API LIVE</span>
        <span class='badge bg-f' style='margin-left:6px;'>
          {'⚡ SPARK' if (SPARK_OK and _spark) else '🐼 PANDAS'}</span>
        <span style='margin-left:8px;'>{pills}</span>
      </div>
      <div class='ptitle'>Project Intelligence</div>
      <div class='psub'>{len(df):,} records · 5 ML objectives · Live from {st.session_state.api_url}</div>
    </div>""",unsafe_allow_html=True)

    findings=[]
    def scan(key,neg=0):
        if key not in M: return None
        try:
            m=M[key]; bf=[c for c in m.get("base",m["features"][:7]) if c in df.columns]
            Xn=df[bf].fillna(0).values
            if m.get("tfidf") and m.get("textf") and "Summary" in df.columns:
                try: Xt=m["tfidf"].transform(df["Summary"].astype(str)).toarray(); Xn=np.hstack([Xn,Xt])
                except: pass
            nin=m["scaler"].n_features_in_; Xs=m["scaler"].transform(Xn[:,:nin])
            nin2=m["model"].n_features_in_ if hasattr(m["model"],"n_features_in_") else Xs.shape[1]
            preds=m["model"].predict(Xs[:,:nin2]); cnt=int((preds==neg).sum()); pct=cnt/len(preds)
            return cnt,pct
        except: return None

    score=100
    def mk(sev,obj,title,action):
        findings.append({"sev":sev,"obj":obj,"title":title,"action":action})
        return 25 if sev=="critical" else 10 if sev=="warning" else 0

    r1=scan("sprint",0); r2=scan("workload",1); r4=scan("burnout",1)
    if r1:
        cnt,pct=r1; sev="critical" if pct>.5 else "warning" if pct>.15 else "ok"
        score-=mk(sev,"Sprint",f"{cnt} sprints at risk ({pct:.0%})" if sev!="ok" else "All sprints on track",
            "Reduce scope or unblock stories." if sev!="ok" else "")
    if r2:
        cnt,pct=r2; sev="critical" if pct>.45 else "warning" if pct>.2 else "ok"
        score-=mk(sev,"Workload",f"{cnt} members overloaded ({pct:.0%})" if sev!="ok" else "All within capacity",
            "Redistribute story points." if sev!="ok" else "")
    if r4:
        cnt,pct=r4; sev="critical" if pct>.5 else "warning" if pct>.25 else "ok"
        score-=mk(sev,"Burnout",f"{cnt} burnout risks ({pct:.0%})" if sev!="ok" else "Team healthy",
            "Lighten next sprint load." if sev!="ok" else "")
    score=max(0,score)
    sc_c="#2d6147" if score>=75 else "#c4973a" if score>=50 else "#c0392b"

    c1,c2,c3,c4,c5,c6=st.columns(6)
    c1.metric("Records",f"{len(df):,}")
    c2.metric("Health",f"{score}/100")
    if "Success_Label" in df.columns: c3.metric("At Risk",f"{int((df['Success_Label']==0).sum()):,}")
    if "Risk_Flag" in df.columns:     c4.metric("Burnout", f"{int(df['Risk_Flag'].sum()):,}")
    c5.metric("Spark Feats",str(len(sf_ok)))
    c6.metric("API Source","Live" if st.session_state.api_ok else "CSV")

    hc1,hc2=st.columns([1,2])
    with hc1:
        st.markdown(f"""<div class='card' style='text-align:center;padding:2.2rem 1rem;'>
          <div style='font-size:0.58rem;text-transform:uppercase;letter-spacing:0.14em;
            color:#8a8070;margin-bottom:10px;font-family:monospace;'>Project Health</div>
          <div style='font-size:4.5rem;font-weight:900;color:{sc_c};
            font-family:"DM Serif Display","Georgia",serif;line-height:1;'>{score}</div>
          <div style='font-size:0.7rem;color:#8a8070;margin-top:5px;'>/100 HEALTH SCORE</div>
          <div class='pw' style='margin:14px 0 8px;'>
            <div class='pf' style='background:{sc_c};width:{score}%;'></div></div>
          <div style='font-size:0.65rem;color:{sc_c};text-transform:uppercase;
            letter-spacing:0.1em;font-family:monospace;'>
            {"● Healthy" if score>=75 else "⚠ Attention" if score>=50 else "✗ At Risk"}</div>
        </div>""",unsafe_allow_html=True)
    with hc2:
        st.markdown("<div class='shdr'>AUTONOMOUS FINDINGS</div>",unsafe_allow_html=True)
        cls={"critical":"red","warning":"yellow","ok":"green"}
        for f in sorted(findings,key=lambda x:{"critical":0,"warning":1,"ok":2}[x["sev"]]):
            act=f"<div style='font-size:0.7rem;color:#4a7c59;margin-top:3px;'>→ {f['action']}</div>" if f["action"] else ""
            st.markdown(f"""<div class='card {cls[f["sev"]]}'>
              <div class='ct'>[{f['obj']}] {f['title']}</div>{act}
            </div>""",unsafe_allow_html=True)

    # Team health
    if "Assignee" in df.columns:
        st.markdown("<div class='shdr'>TEAM HEALTH</div>",unsafe_allow_html=True)
        people=sorted(df["Assignee"].unique()); cols_a=st.columns(min(len(people),8))
        for i,p in enumerate(people):
            sub=df[df["Assignee"]==p]
            sr=sub["Success_Label"].eq(0).mean() if "Success_Label" in df.columns else 0
            ol=sub["Expected_Overload"].mean()    if "Expected_Overload" in df.columns else 0
            br=sub["Risk_Flag"].mean()             if "Risk_Flag" in df.columns else 0
            wl=sub["Current_Workload_Percent"].mean() if "Current_Workload_Percent" in df.columns else 0
            ps=max(0,min(100,100-(sr*35)-(ol*30)-(br*20)-max(0,(wl-100)/2)))
            pc="#2d6147" if ps>=60 else "#c4973a" if ps>=40 else "#c0392b"
            with cols_a[i%len(cols_a)]:
                st.markdown(f"""<div class='acard'>
                  <div class='av' style='background:{pc}20;color:{pc};'>{p[0]}</div>
                  <div class='an'>{p}</div>
                  <div class='as' style='color:{pc};margin:3px 0;'>{ps:.0f}</div>
                  <div class='ad'>WL {wl:.0f}% · Burn {br:.0%}</div>
                </div>""",unsafe_allow_html=True)

    # Velocity from API
    if st.session_state.velocity:
        vd=st.session_state.velocity.get("sprints",[])
        if vd:
            st.markdown("<div class='shdr'>SPRINT VELOCITY (FROM API)</div>",unsafe_allow_html=True)
            vdf=pd.DataFrame(vd)
            if "name" in vdf.columns: vdf=vdf.set_index("name")
            elif "sprint_name" in vdf.columns: vdf=vdf.set_index("sprint_name")
            c1,c2=st.columns(2)
            with c1:
                st.caption("Velocity per Sprint")
                vcols=[c for c in ["velocity","completed"] if c in vdf.columns]
                if vcols: st.bar_chart(vdf[vcols],height=180)
            with c2:
                st.caption("Completion %")
                pcols=[c for c in ["pct","percent_done","pct_done"] if c in vdf.columns]
                if pcols: st.line_chart(vdf[[pcols[0]]],height=180)

    # Trends
    if "Sprint_Number" in df.columns:
        st.markdown("<div class='shdr'>TRENDS</div>",unsafe_allow_html=True)
        tr=df.copy(); tr["risk"]=(tr["Success_Label"]==0).astype(int)
        agg=tr.groupby("Sprint_Number").agg(
            risk=("risk","mean"),wl=("Current_Workload_Percent","mean"),
            burn=("Risk_Flag","mean")).reset_index()
        tc1,tc2,tc3=st.columns(3)
        with tc1: st.caption("Sprint Risk %"); st.line_chart((agg.set_index("Sprint_Number")[["risk"]]*100).round(1),height=150)
        with tc2: st.caption("Avg Workload %"); st.line_chart(agg.set_index("Sprint_Number")[["wl"]].round(1),height=150)
        with tc3: st.caption("Burnout %");      st.line_chart((agg.set_index("Sprint_Number")[["burn"]]*100).round(1),height=150)

    st.download_button("⬇ EXPORT REPORT",
        data=f"# Agile Intelligence Report\nHealth: {score}/100\n\n"+
             "\n".join([f"- [{f['obj']}] {f['title']}" for f in findings]),
        file_name="report.md",mime="text/markdown")

# ══════════════════════════════════════════════════════════════════════════
# Shared pred card helper
def pred_card(label,value,conf_str,color):
    st.markdown(f"""<div class='pred'>
      <div class='pred-lbl'>{label}</div>
      <div class='pred-val' style='color:{color};'>{value}</div>
      <div class='pred-conf'>{conf_str}</div>
    </div>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SPRINT
# ══════════════════════════════════════════════════════════════════════════
with T[2]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Sprint Completion</div>
      <div class='psub'>Fine-tuned Logistic Regression + TF-IDF</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if "sprint" not in M: st.warning("Sprint model unavailable.")
    else:
        m=M["sprint"]; acc=m["acc"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Accuracy",f"{acc:.2%}",f"{acc-0.89:+.2%} vs 89%")
        c2.metric("F1 Score",f"{m['f1']:.3f}")
        c3.metric("AUC-ROC",f"{m['auc']:.3f}")
        c4.metric("Text Feats",str(len(m.get("textf",[]))))
        bc="#2d6147" if acc>=0.89 else "#c4973a"
        st.markdown(f"""<div style='margin:0.5rem 0 1rem;'>
          <div style='display:flex;justify-content:space-between;font-size:0.68rem;color:#8a8070;'>
            <span>Current: {acc:.2%}</span><span>Target: 89%</span></div>
          <div class='pw'><div class='pf' style='background:{bc};width:{acc*100:.1f}%;'></div></div>
          <div class='pw' style='margin-top:3px;'>
            <div class='pf' style='background:#ddd5c4;width:89%;'></div></div>
        </div>""",unsafe_allow_html=True)
        cl,cr=st.columns([3,2])
        with cl:
            with st.expander("◈ Fine-tune Results"):
                if m.get("tune"):
                    td=pd.DataFrame(m["tune"]).sort_values("acc",ascending=False)
                    td["acc"]=td["acc"].apply(lambda x:f"{x:.2%}"); td["f1"]=td["f1"].apply(lambda x:f"{x:.3f}")
                    st.dataframe(td[["C","solver","class_weight","max_iter","acc","f1"]],use_container_width=True,hide_index=True)
            with st.expander("◈ Classification Report"): st.text(m["report"])
            with st.expander("◈ Confusion Matrix"):
                cm=np.array(m["cm"])
                st.dataframe(pd.DataFrame(cm,index=["Actual: Risk","Actual: Done"],
                    columns=["Pred: Risk","Pred: Done"]),use_container_width=True)
        with cr:
            st.markdown("<div class='shdr'>PREDICT</div>",unsafe_allow_html=True)
            psp=st.number_input("Planned SP",    1,100,40,key="s1")
            csp=st.number_input("Completed SP",  0,100,30,key="s2")
            pct=st.slider("% Done",0.0,100.0,75.0,       key="s3")
            drs=st.number_input("Days Left",     0, 30, 5,key="s4")
            hv =st.number_input("Hist Velocity", 0,100,35,key="s5")
            bs =st.number_input("Blocked",       0, 10, 1,key="s6")
            sc2=st.number_input("Scope Change", -20,20, 0,key="s7")
            smr=st.text_input("Summary","Fix login bug",  key="s8")
            if st.button("◆ PREDICT SPRINT",key="s_btn",use_container_width=True):
                rv=[{"Planned_Story_Points_Sprint":psp,"Completed_Story_Points":csp,
                     "Percent_Done":pct,"Days_Remaining_Sprint":drs,"Historical_Velocity":hv,
                     "Blocked_Stories":bs,"Scope_Change":sc2}.get(f,0) for f in m.get("base",m["features"][:7])]
                if m.get("tfidf") and m.get("textf"):
                    try: rv+=list(m["tfidf"].transform([smr.lower()]).toarray()[0])
                    except: pass
                ra=np.array(rv).reshape(1,-1); nin=m["scaler"].n_features_in_
                rs=m["scaler"].transform(ra[:,:nin])
                nin2=m["model"].n_features_in_ if hasattr(m["model"],"n_features_in_") else rs.shape[1]
                p=m["model"].predict(rs[:,:nin2])[0]; pb=m["model"].predict_proba(rs[:,:nin2])[0][1]
                pc="#2d6147" if pb>=0.65 else "#c4973a" if pb>=0.4 else "#c0392b"
                pred_card("Sprint Prediction","COMPLETE ✓" if p else "AT RISK ✗",f"{pb:.2%} confidence",pc)

# ══════════════════════════════════════════════════════════════════════════
# TABs 3-6 — Workload, TTR, Burnout, Allocation
# ══════════════════════════════════════════════════════════════════════════
with T[3]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Workload Projection</div><div class='psub'>Naive Bayes · binary overload classifier</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if "workload" not in M: st.warning("Workload model unavailable.")
    else:
        m=M["workload"]
        c1,c2,c3=st.columns(3)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("F1",f"{m['f1']:.3f}"); c3.metric("Algo",m["algo"])
        with st.expander("◈ Report"): st.text(m["report"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='shdr'>PREDICT</div>",unsafe_allow_html=True)
            psp2=st.number_input("Planned SP",         1,100, 35,key="w1")
            casp=st.number_input("Current Assigned SP",0,100, 40,key="w2")
            hasp=st.number_input("Historical Avg SP",  1,100, 30,key="w3")
            rdr =st.number_input("Remaining Days",     1, 30,  5,key="w4")
            hpt =st.number_input("High Priority",      0, 10,  2,key="w5")
            cwp =st.number_input("Workload %",         0,200,125,key="w6")
            if st.button("◆ PREDICT OVERLOAD",key="w_btn",use_container_width=True):
                row=pd.DataFrame([{"Planned_Story_Points_Resource":psp2,"Current_Assigned_SP":casp,
                    "Historical_Avg_SP":hasp,"Remaining_Days_Resource":rdr,
                    "High_Priority_Tasks_Resource":hpt,"Current_Workload_Percent":cwp}])
                row=row.reindex(columns=m["features"],fill_value=0)
                pred=m["model"].predict(m["scaler"].transform(row))[0]
                prob=m["model"].predict_proba(m["scaler"].transform(row))[0][1]
                pc="#c0392b" if pred else "#2d6147"
                pred_card("Workload Status","OVERLOADED" if pred else "WITHIN CAPACITY",f"{prob:.2%} confidence",pc)

with T[4]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Time to Resolve</div><div class='psub'>Ridge Regression · regularised hour prediction</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if "ttr" not in M: st.warning("TTR model unavailable.")
    else:
        m=M["ttr"]
        c1,c2,c3=st.columns(3)
        c1.metric("R²",f"{m['r2']:.3f}"); c2.metric("MSE",f"{m['mse']:.2f}"); c3.metric("Algo",m["algo"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='shdr'>ESTIMATE</div>",unsafe_allow_html=True)
            itype=st.selectbox("Type",["Bug","Story","Task"],key="t1")
            pri  =st.selectbox("Priority",["Low","Medium","High"],key="t2")
            oe   =st.number_input("Estimate (h)",1,50,8,key="t3")
            sp   =st.number_input("Story Points",1,20,5,key="t4")
            if st.button("◆ ESTIMATE TTR",key="t_btn",use_container_width=True):
                row={c:0 for c in m["features"]}
                if f"Issue_Type_{itype}" in row: row[f"Issue_Type_{itype}"]=1
                if f"Priority_{pri}"     in row: row[f"Priority_{pri}"]    =1
                row["Original_Estimate_Hours"]=oe
                if "Story_Points_Issue" in row: row["Story_Points_Issue"]=sp
                pt=max(0,m["model"].predict(m["scaler"].transform(pd.DataFrame([row])[m["features"]]))[0])
                delta=pt-oe; pc="#2d6147" if delta<=2 else "#c4973a" if delta<=5 else "#c0392b"
                pred_card("Estimated Time",f"{pt:.1f} hours",f"{delta:+.1f}h vs original estimate",pc)

with T[5]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Burnout Risk</div><div class='psub'>Decision Tree · explainable threshold rules</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if "burnout" not in M: st.warning("Burnout model unavailable.")
    else:
        m=M["burnout"]
        c1,c2,c3=st.columns(3)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("F1",f"{m['f1']:.3f}"); c3.metric("Algo",m["algo"])
        with st.expander("◈ Report"): st.text(m["report"])
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='shdr'>CHECK RISK</div>",unsafe_allow_html=True)
            tsp =st.number_input("Total SP",         0,100,40,key="b1")
            hasp=st.number_input("Historical Avg SP",1,100,25,key="b2")
            hpt =st.number_input("High Priority",    0, 10, 2,key="b3")
            co  =st.number_input("Consec. Overloads",0,  5, 2,key="b4")
            if st.button("◆ CHECK BURNOUT",key="b_btn",use_container_width=True):
                row={f:0 for f in m["features"]}
                row.update({"Total_SP_This_Sprint":tsp,"Historical_Avg_SP_Burnout":hasp,
                            "High_Priority_Tasks_Burnout":hpt,"Consecutive_Overloads":co})
                rs=m["scaler"].transform(pd.DataFrame([row])[m["features"]])
                pred=m["model"].predict(rs)[0]; prob=m["model"].predict_proba(rs)[0][1]
                pc="#c0392b" if pred else "#2d6147"
                pred_card("Burnout Status","RISK DETECTED" if pred else "HEALTHY",f"{prob:.2%} confidence",pc)

with T[6]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Resource Allocation</div><div class='psub'>KNN + TF-IDF · issue-to-assignee matching</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if "alloc" not in M: st.warning("Allocation model unavailable.")
    else:
        m=M["alloc"]
        c1,c2=st.columns(2)
        c1.metric("Accuracy",f"{m['acc']:.2%}"); c2.metric("Algo",m["algo"])
        st.info("ℹ️ Accuracy improves with real Jira skill-tag and component features.")
        cl,cr=st.columns([3,2])
        with cr:
            st.markdown("<div class='shdr'>SUGGEST</div>",unsafe_allow_html=True)
            smry=st.text_input("Summary","Fix login bug",key="a1")
            lbl2=st.text_input("Label",  "Bug",          key="a2")
            oe5 =st.number_input("Estimate (h)",1,50,8,  key="a3")
            sp5 =st.number_input("Story Points",1,20,5,  key="a4")
            if st.button("◆ SUGGEST ASSIGNEE",key="a_btn",use_container_width=True):
                try:    se=m["le_s"].transform([smry])[0]
                except: se=0
                try:    le=m["le_l"].transform([lbl2])[0]
                except: le=0
                row=pd.DataFrame([{"Summary_enc":se,"Labels_enc":le,
                    "Original_Estimate_Resource":oe5,"Story_Points_Resource":sp5}])
                row=row.reindex(columns=m["features"],fill_value=0)
                asgn=m["model"].predict(m["scaler"].transform(row))[0]
                pred_card("Recommended Assignee",str(asgn),"Based on similar past issues","#2d6147")

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — BERT TEXT
# ══════════════════════════════════════════════════════════════════════════
with T[7]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>BERT Text Intelligence</div>
      <div class='psub'>TF-IDF text classification + keyword extraction · install transformers for full BERT</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    cat_c={"Bug":"#c0392b","Feature":"#2d6147","Tech-Debt":"#c4973a",
            "Performance":"#1a3c2e","Security":"#5c4a3a","Regression":"#8a4a2e","Other":"#8a8070"}
    if "Summary" in df.columns and tp:
        c1,c2=st.columns([1,2])
        with c1:
            st.markdown("<div class='shdr'>CATEGORIES</div>",unsafe_allow_html=True)
            vc=pd.Series(tp).value_counts()
            for cat,cnt in vc.items():
                p=cnt/len(tp); col=cat_c.get(cat,"#8a8070")
                st.markdown(f"""<div style='margin-bottom:7px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.73rem;'>
                    <span style='color:{col};font-weight:700;font-family:monospace;'>{cat}</span>
                    <span style='color:#8a8070;'>{cnt} ({p:.0%})</span></div>
                  <div class='pw'><div class='pf' style='background:{col};width:{p*100:.1f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='shdr'>TOP KEYWORDS</div>",unsafe_allow_html=True)
            for cat,terms in list(tt.items())[:6]:
                col=cat_c.get(cat,"#8a8070")
                badges="".join([f"<span style='background:{col}18;color:{col};border:1px solid {col}28;padding:1px 8px;border-radius:2px;font-size:0.62rem;font-weight:700;margin:2px;display:inline-block;font-family:monospace;'>{t}</span>" for t in terms])
                st.markdown(f"<div style='margin-bottom:9px;'><b style='color:{col};font-size:0.76rem;font-family:monospace;'>{cat}</b><br>{badges}</div>",unsafe_allow_html=True)
        st.markdown("<div class='shdr'>CLASSIFIED ISSUES</div>",unsafe_allow_html=True)
        pv=df[["Summary"]].copy()
        for c in ["Priority","Assignee","Issue_Type","Status"]:
            if c in df.columns: pv[c]=df[c]
        pv["BERT_Category"]=tp
        st.dataframe(pv.head(20),use_container_width=True,hide_index=True)
        st.markdown("<div class='shdr'>CLASSIFY NEW TEXT</div>",unsafe_allow_html=True)
        cl2,cr2=st.columns([3,2])
        with cr2:
            ntxt=st.text_area("Issue summary","Fix null pointer exception in auth",key="bert_in")
            if st.button("◆ CLASSIFY",key="bert_btn",use_container_width=True):
                p2,_=bert_classify(tuple([ntxt])); cat=p2[0] if p2 else "Unknown"
                col=cat_c.get(cat,"#8a8070")
                pred_card("BERT Category",cat,"TF-IDF text classification",col)
    else:
        st.info("Dataset needs a 'Summary' column.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — SPARK ML
# ══════════════════════════════════════════════════════════════════════════
with T[8]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Spark ML Engine</div>
      <div class='psub'>Ensemble Voting · GBT · Improved K-Means++ · Anomaly Detection</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if SPARK_OK and _spark: st.success("⚡ Apache Spark active")
    else: st.info("🐼 Pandas mode — install Java 17 + pyspark for Spark")
    if sf_ok:
        pills="".join([f"<span class='spill'>⚡{f}</span>" for f in sf_ok])
        st.markdown(f"<div style='margin-bottom:1rem;'>{pills}</div>",unsafe_allow_html=True)

    st.markdown("<div class='shdr'>SPRINT ENSEMBLE (LR + GBT + RF + ADABOOST)</div>",unsafe_allow_html=True)
    if "sprint" in SM:
        s=SM["sprint"]; ea=s["acc"]; best=max(s["ind"],key=s["ind"].get)
        c1,c2,c3=st.columns(3)
        c1.metric("Ensemble Acc",f"{ea:.2%}"); c2.metric("Best Single",f"{s['ind'][best]:.2%}",best)
        c3.metric("Spark Feats",str(len([f for f in s["feat"] if f in s["sf"]])))
        with st.expander("◈ Comparison + Feature Importance"):
            for nm,acc in sorted(s["ind"].items(),key=lambda x:-x[1]):
                diff=ea-acc; dc="#2d6147" if diff>=0 else "#c0392b"
                st.markdown(f"""<div style='margin-bottom:5px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.76rem;'>
                    <span style='color:#1a2420;'>{nm}</span>
                    <span>{acc:.2%} <span style='color:{dc};'>({diff:+.2%})</span></span></div>
                  <div class='pw'><div class='pf' style='background:#4a7c59;width:{acc*100:.1f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
            st.markdown("**GBT Feature Importance:**")
            imp=pd.Series(s["imp"]).sort_values(ascending=False)
            for feat,v in imp.head(10).items():
                bw=v/imp.max()*100; tag=" ⚡" if feat in s["sf"] else ""
                col="#2d6147" if feat in s["sf"] else "#4a7c59"
                st.markdown(f"""<div style='margin-bottom:3px;'>
                  <div style='display:flex;justify-content:space-between;font-size:0.7rem;color:#1a2420;'>
                    <span style='font-family:monospace;'>{feat}{tag}</span>
                    <span style='color:{col};'>{v:.3f}</span></div>
                  <div class='pw'><div class='pf' style='background:{col};width:{bw:.0f}%;'></div></div>
                </div>""",unsafe_allow_html=True)
    else: st.info("Sprint ensemble unavailable.")

    st.markdown("<div class='shdr'>TTR — GBT REGRESSOR</div>",unsafe_allow_html=True)
    if "ttr" in SM:
        t=SM["ttr"]
        c1,c2,c3,c4=st.columns(4)
        c1.metric("GBT R²",f"{t['gb_r2']:.3f}"); c2.metric("LR R²",f"{t['lr_r2']:.3f}",f"{t['gb_r2']-t['lr_r2']:+.3f}")
        c3.metric("GBT MSE",f"{t['gb_mse']:.2f}"); c4.metric("Spark Feats",str(len([f for f in t["feat"] if f in t["sf"]])))
    else: st.info("TTR unavailable.")

    st.markdown("<div class='shdr'>BURNOUT — GBT + RF ENSEMBLE</div>",unsafe_allow_html=True)
    if "burnout" in SM:
        b=SM["burnout"]
        c1,c2=st.columns(2); c1.metric("Acc",f"{b['acc']:.2%}"); c2.metric("Spark Feats",str(len([f for f in b["feat"] if f in b["sf"]])))
        with st.expander("◈ Report"): st.text(b["report"])
    else: st.info("Burnout unavailable.")

    st.markdown("<div class='shdr'>TEAM CLUSTERING — K-MEANS++ ELBOW</div>",unsafe_allow_html=True)
    if "cluster" in SM:
        cl=SM["cluster"]
        ec1,ec2=st.columns(2)
        with ec1: st.caption("Elbow Method"); st.line_chart(pd.DataFrame({"k":cl["k_range"],"Inertia":cl["inertias"]}).set_index("k"),height=180)
        with ec2: st.caption("Silhouette Score"); st.line_chart(pd.DataFrame({"k":cl["k_range"],"Silhouette":cl["sils"]}).set_index("k"),height=180)
        kc1,kc2,kc3=st.columns(3)
        kc1.metric("Optimal k",cl["best_k"]); kc2.metric("Silhouette",f"{cl['sil']:.3f}"); kc3.metric("Init","K-Means++")
        adf=pd.DataFrame(cl["agg"]); nc=cl["num_cols"]
        cl1,cl2,cl3=st.columns(3)
        for cw,cid,lbl,col in [(cl1,0,"High Performers","#2d6147"),(cl2,1,"Mid Performers","#c4973a"),(cl3,2,"Overloaded","#c0392b")]:
            members=adf[adf["Cluster"]==cid].index.tolist() if cid<cl["best_k"] else []
            with cw:
                badges="".join([f"<span style='background:{col}18;color:{col};border:1px solid {col}28;padding:1px 8px;border-radius:2px;font-size:0.7rem;font-weight:700;margin:2px;display:inline-block;font-family:monospace;'>{m}</span>" for m in members])
                st.markdown(f"""<div class='card' style='text-align:center;'>
                  <div style='color:{col};font-weight:700;font-size:0.75rem;margin-bottom:6px;font-family:monospace;'>{lbl}</div>{badges}
                </div>""",unsafe_allow_html=True)
        ddf=adf.drop(columns=["Cluster"],errors="ignore")
        fmt={c:"{:.2f}" for c in nc if c in ddf.columns}
        st.dataframe(ddf.style.format(fmt,na_rep="--"),use_container_width=True)
    else: st.info("Clustering unavailable.")

    st.markdown("<div class='shdr'>ANOMALY DETECTION — ISOLATION FOREST</div>",unsafe_allow_html=True)
    if "anomaly" in SM:
        an=SM["anomaly"]
        sc_s=pd.Series(an["scores"]); cf_s=pd.Series(an["confs"])
        mask=sc_s==-1; adf2=df[mask].copy(); adf2["Score"]=cf_s[mask].values
        c1,c2,c3=st.columns(3); c1.metric("Anomalies",an["count"]); c2.metric("Rate","5%"); c3.metric("Features",len(an["feats"]))
        dcols=(["Assignee"] if "Assignee" in df.columns else [])+an["feats"]+["Score"]
        st.dataframe(adf2[[c for c in dcols if c in adf2.columns]].head(10),use_container_width=True,hide_index=True)
    else: st.info("Anomaly detection unavailable.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — EVAL
# ══════════════════════════════════════════════════════════════════════════
with T[9]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Model Evaluation</div>
      <div class='psub'>Accuracy · F1 · AUC-ROC · Confusion Matrices · 89% target · Fine-tune</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    rows=[]
    for key,label in [("sprint","Sprint"),("workload","Workload"),("burnout","Burnout"),("alloc","Allocation")]:
        if key in M:
            m=M[key]; acc=m.get("acc",0)
            rows.append({"Objective":label,"Algorithm":m["algo"],"Accuracy":f"{acc:.2%}",
                "F1":f"{m.get('f1',0):.3f}",
                "AUC":f"{m.get('auc',0):.3f}" if isinstance(m.get("auc"),float) else "—",
                "vs 89%":f"{acc-0.89:+.2%} {'✓' if acc>=0.89 else '✗'}"})
    if "ttr" in M:
        m=M["ttr"]; rows.append({"Objective":"TTR","Algorithm":m["algo"],"Accuracy":f"R²={m['r2']:.3f}","F1":"—","AUC":"—","vs 89%":"—"})
    if rows:
        st.markdown("<div class='shdr'>PERFORMANCE SUMMARY</div>",unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("<div class='shdr'>ACCURACY VS 89% TARGET</div>",unsafe_allow_html=True)
    for key,label in [("sprint","Sprint"),("workload","Workload"),("burnout","Burnout"),("alloc","Allocation")]:
        if key in M:
            acc=M[key].get("acc",0); col="#2d6147" if acc>=0.89 else "#c4973a" if acc>=0.75 else "#c0392b"
            st.markdown(f"""<div style='margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;font-size:0.78rem;'>
                <span><b style='color:#1a2420;'>{label}</b> <span style='color:#8a8070;font-family:monospace;font-size:0.7rem;'>{M[key]['algo']}</span></span>
                <span style='color:{col};font-family:monospace;'>{acc:.2%} {"✓" if acc>=0.89 else "✗"}</span></div>
              <div class='pw'><div class='pf' style='background:{col};width:{acc*100:.1f}%;'></div></div>
            </div>""",unsafe_allow_html=True)
    if "sprint" in M and M["sprint"].get("tune"):
        st.markdown("<div class='shdr'>SPRINT FINE-TUNE TABLE</div>",unsafe_allow_html=True)
        tdf=pd.DataFrame(M["sprint"]["tune"]).sort_values("acc",ascending=False)
        tdf["acc"]=tdf["acc"].apply(lambda x:f"{x:.2%}"); tdf["f1"]=tdf["f1"].apply(lambda x:f"{x:.3f}")
        st.dataframe(tdf[["C","solver","class_weight","max_iter","acc","f1"]],use_container_width=True,hide_index=True)
    st.markdown("<div class='shdr'>CONFUSION MATRICES</div>",unsafe_allow_html=True)
    cm_cols=st.columns(2); ci=0
    for key,label,names in [("sprint","Sprint",["Risk","Complete"]),("workload","Workload",["No OL","Overload"]),("burnout","Burnout",["Healthy","At Risk"])]:
        if key in M and M[key].get("cm"):
            cm=np.array(M[key]["cm"])
            with cm_cols[ci%2]:
                st.caption(label)
                st.dataframe(pd.DataFrame(cm,index=[f"Actual: {n}" for n in names],columns=[f"Pred: {n}" for n in names]),use_container_width=True)
            ci+=1

# ══════════════════════════════════════════════════════════════════════════
# TAB 10 — WRITE
# ══════════════════════════════════════════════════════════════════════════
with T[10]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Write to API</div>
      <div class='psub'>Create issues · Add comments · Transition status · Search</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    if not st.session_state.api_ok:
        st.markdown("""<div class='card forest' style='text-align:center;padding:3rem;'>
          <div style='font-family:"DM Serif Display","Georgia",serif;font-size:2rem;color:#ddd5c4;margin-bottom:8px;'>◆</div>
          <div style='font-size:0.82rem;color:#8a8070;'>Connect in <b style='color:#2d6147;'>◆ CONNECT</b> tab first</div>
        </div>""",unsafe_allow_html=True)
    else:
        client=st.session_state.client
        st.success(f"✓ Connected to {st.session_state.api_url}")
        wt1,wt2,wt3,wt4=st.tabs(["CREATE ISSUE","ADD COMMENT","TRANSITION","SEARCH"])
        with wt1:
            c1,c2=st.columns(2)
            with c1:
                ni_s=st.text_input("Summary","",key="ni_s")
                ni_t=st.selectbox("Type",["Task","Bug","Story","Epic"],key="ni_t")
                ni_p=st.selectbox("Priority",["High","Medium","Low"],key="ni_p")
            with c2:
                ni_a=st.selectbox("Assignee",["Unassigned"]+["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry"],key="ni_a")
                ni_sp=st.number_input("Story Points",1,21,3,key="ni_sp")
                ni_spr=st.text_input("Sprint ID","SPR-08",key="ni_spr")
                ni_l=st.text_input("Labels","general",key="ni_l")
            if st.button("◆ CREATE",key="ci_btn"):
                if not ni_s: st.warning("Summary required.")
                else:
                    d,e=client.create_issue(ni_s,ni_t,ni_p,ni_a,ni_sp,ni_l)
                    if e: st.error(e)
                    else: st.success(f"✓ Created: **{d.get('issue_id','?')}**")
        with wt2:
            ck=st.text_input("Issue Key","AGI-0001",key="ck"); ct=st.text_area("Comment","",height=80,key="ct")
            if st.button("◆ ADD COMMENT",key="cmt_btn"):
                d,e=client.add_comment(ck,ct)
                if e: st.error(e)
                else: st.success("✓ Comment added")
        with wt3:
            tk=st.text_input("Issue Key","AGI-0001",key="tk")
            tto=st.selectbox("Transition To",["In Progress","In Review","Done","Blocked","To Do"],key="tto")
            if st.button("◆ TRANSITION",key="tr_btn"):
                d,e=client.transition(tk,tto)
                if e: st.error(e)
                else: st.success(f"✓ → {tto}")
        with wt4:
            c1,c2,c3,c4=st.columns(4)
            with c1: sq=st.text_input("Keyword","",key="sq")
            with c2: sa=st.selectbox("Assignee",["(any)"]+["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry","Iris","Jack","Karen","Leo","Maya","Nora","Oscar"],key="sa")
            with c3: ss=st.selectbox("Status",["(any)","To Do","In Progress","Done","Blocked"],key="ss")
            with c4: st_team=st.selectbox("Team",["(any)"]+["Platform","Backend","Frontend","Mobile","Data","Infrastructure","Security","QA","DevOps","Product"],key="st_team")
            if st.button("◆ SEARCH",key="srch"):
                results,e=client.search(sq,sa if sa!="(any)" else None,None,ss if ss!="(any)" else None)
                if e: st.error(e)
                elif not results: st.info("No results.")
                else:
                    rdf=pd.DataFrame(results); st.success(f"✓ {len(rdf)} results")
                    cols=[c for c in ["Issue_ID","Summary","Issue_Type","Priority","Status","Assignee","Story_Points_Issue"] if c in rdf.columns]
                    st.dataframe(rdf[cols],use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 11 — LIVE DATA
# ══════════════════════════════════════════════════════════════════════════
with T[11]:
    st.markdown("""<div style='padding:2rem 2.5rem 0;'>
      <div class='ptitle'>Live Data Explorer</div>
      <div class='psub'>Browse raw data fetched from the API — sprints, team, issues, analytics</div>
    </div><div style='padding:0 2.5rem;'>""",unsafe_allow_html=True)
    dt1,dt2,dt3,dt4=st.tabs(["SPRINTS","TEAM","ISSUES","ANALYTICS"])
    with dt1:
        if st.session_state.sprints:
            sdf=pd.DataFrame(st.session_state.sprints)
            c1,c2=st.columns(2)
            c1.metric("Active",sum(1 for s in st.session_state.sprints if s["state"]=="active"))
            c2.metric("At Risk",sum(1 for s in st.session_state.sprints if not s["success"]))
            cols=[c for c in ["sprint_id","sprint_name","state","start_date","end_date","planned_sp","completed_sp","percent_done","historical_velocity","blocked_stories","success","risk_flag"] if c in sdf.columns]
            st.dataframe(sdf[cols].round(1),use_container_width=True,hide_index=True)
        else: st.info("Connect API to see sprints.")
    with dt2:
        if st.session_state.team:
            tdf=pd.DataFrame(st.session_state.team)
            cols=[c for c in ["name","role","team","total_issues","completion_rate","workload_pct","burnout_score","health_score","consecutive_ol","risk_flag"] if c in tdf.columns]
            st.dataframe(tdf[cols].round(1),use_container_width=True,hide_index=True)
            bc1,bc2=st.columns(2)
            with bc1:
                st.caption("Burnout Score")
                if "burnout_score" in tdf.columns: st.bar_chart(tdf.set_index("name")[["burnout_score"]].head(20),height=200)
            with bc2:
                st.caption("Workload %")
                if "workload_pct" in tdf.columns: st.bar_chart(tdf.set_index("name")[["workload_pct"]].head(20),height=200)
        else: st.info("Connect API to see team.")
    with dt3:
        show_cols=[c for c in ["Issue_ID","Summary","Issue_Type","Priority","Status","Assignee","Story_Points_Issue","Resolution_Time_Hours"] if c in df.columns]
        st.dataframe(df[show_cols].head(30) if show_cols else df.head(30),use_container_width=True,hide_index=True)
    with dt4:
        if st.session_state.summary:
            h=st.session_state.summary
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Health",f"{h.get('health_score',0):.0f}/100")
            c2.metric("At Risk Sprints",h.get("sprints_at_risk",0))
            c3.metric("Burnout Risks",h.get("burnout_risk",0))
            c4.metric("Avg Velocity",h.get("avg_velocity",0))
            st.json(h)
        else: st.info("Connect API to see analytics.")