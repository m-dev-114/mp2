"""
╔══════════════════════════════════════════════════════════════╗
║  AGILE INTELLIGENCE  —  DATA WEBSITE                         ║
║  FastAPI · 50 Employees · 2 Months · REST API                ║
║  Run: uvicorn agile_api_server:app --reload --port 8000      ║
╚══════════════════════════════════════════════════════════════╝
"""
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np, random, math
from datetime import datetime, timedelta, date

app = FastAPI(title="Agile Intelligence Data Platform", version="3.0.0",
              docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════════════════════
# 50 EMPLOYEES  +  2 MONTHS OF REALISTIC DATA
# ══════════════════════════════════════════════════════════════════════════
FIRST = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Henry",
         "Iris","Jack","Karen","Leo","Maya","Nora","Oscar","Pam",
         "Quinn","Ryan","Sara","Tom","Uma","Victor","Wendy","Xander",
         "Yara","Zane","Aiden","Bella","Chris","Diana","Ethan","Fiona",
         "George","Hannah","Ivan","Julia","Kyle","Luna","Marcus","Nina",
         "Oliver","Petra","Raj","Sofia","Tyler","Ursula","Vince","Willa",
         "Xerxes","Yasmin"]

LAST  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller",
         "Davis","Wilson","Taylor","Anderson","Thomas","Jackson","White",
         "Harris","Martin","Thompson","Robinson","Clark","Lewis","Lee",
         "Walker","Hall","Allen","Young","King","Wright","Scott","Green",
         "Baker","Adams","Nelson","Hill","Ramirez","Campbell","Mitchell",
         "Roberts","Carter","Phillips","Evans","Turner","Torres","Parker",
         "Collins","Edwards","Stewart","Morris","Rogers","Reed","Cook"]

ROLES  = ["Senior Engineer","Software Engineer","Junior Engineer",
          "Tech Lead","QA Engineer","DevOps Engineer","Data Engineer",
          "Product Manager","Scrum Master","UX Designer"]

TEAMS  = ["Platform","Backend","Frontend","Mobile","Data","Infrastructure",
          "Security","QA","DevOps","Product"]

ISSUE_TYPES = ["Bug","Story","Task","Epic","Sub-task"]
PRIORITIES  = ["Critical","High","Medium","Low"]
STATUSES    = ["To Do","In Progress","In Review","Done","Blocked"]
LABELS      = ["feature","bug","tech-debt","hotfix","regression",
               "security","performance","documentation","testing","infrastructure"]
SUMMARIES   = [
    "Fix authentication session timeout","Implement OAuth 2.0 login",
    "Refactor database connection pooling","Add Redis caching layer",
    "Fix memory leak in background worker","Deploy Kubernetes cluster upgrade",
    "Add end-to-end test coverage","Migrate legacy API to REST",
    "Fix CORS policy misconfiguration","Implement rate limiting middleware",
    "Add real-time notification system","Optimize slow database queries",
    "Fix null pointer in payment service","Security audit and penetration test",
    "Upgrade dependency versions","Add dark mode to dashboard",
    "Implement data export feature","Fix mobile layout responsiveness",
    "Add two-factor authentication","Refactor monolith to microservices",
    "Fix race condition in scheduler","Implement GraphQL subscriptions",
    "Add monitoring and alerting","Fix CSV import validation bug",
    "Implement webhook retry logic","Add search functionality to admin",
    "Fix timezone handling bug","Implement audit logging",
    "Add load balancer health checks","Fix email template rendering",
    "Implement feature flag system","Add API documentation with Swagger",
    "Fix Docker container networking","Optimize CI/CD pipeline speed",
    "Implement blue-green deployment","Add database backup automation",
    "Fix XSS vulnerability in editor","Add pagination to list endpoints",
    "Implement data archival policy","Fix broken PDF generation",
]

# ── Two-month date range (8 sprints of ~1 week each) ─────────────────────
TODAY      = date.today()
START_DATE = TODAY - timedelta(days=60)

def date_range_days():
    """Returns list of all dates in our 2-month window."""
    return [START_DATE + timedelta(days=i) for i in range(61)]

def sprint_dates(sprint_num):
    """Each sprint = ~7.5 days, 8 sprints over 60 days."""
    sprint_start = START_DATE + timedelta(days=(sprint_num-1)*7)
    sprint_end   = sprint_start + timedelta(days=7)
    return sprint_start.strftime("%Y-%m-%d"), sprint_end.strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════════════════════════
# DATA GENERATORS
# ══════════════════════════════════════════════════════════════════════════
def build_employees(seed=42):
    rng = np.random.default_rng(seed)
    employees = []
    used_names = set()
    for i in range(50):
        while True:
            fn = FIRST[i % len(FIRST)]
            ln = rng.choice(LAST)
            name = f"{fn} {ln}"
            if name not in used_names:
                used_names.add(name); break
        role    = str(rng.choice(ROLES))
        team    = str(rng.choice(TEAMS))
        exp     = int(rng.integers(1, 12))
        base_sp = float(rng.integers(20, 70))
        employees.append({
            "employee_id":   f"EMP-{i+1:03d}",
            "name":          name,
            "first_name":    fn,
            "role":          role,
            "team":          team,
            "experience_yrs":exp,
            "base_velocity": base_sp,
            "email":         f"{fn.lower()}.{ln.lower()}@agile.dev",
            "joined":        (START_DATE - timedelta(days=int(rng.integers(30,1000)))).strftime("%Y-%m-%d"),
        })
    return employees

def build_sprints(seed=42):
    rng = np.random.default_rng(seed)
    sprints = []
    for i in range(1, 9):  # 8 sprints over 2 months
        s_start, s_end = sprint_dates(i)
        end_dt   = date.fromisoformat(s_end)
        is_done  = end_dt <= TODAY
        is_active= (date.fromisoformat(s_start) <= TODAY <= end_dt)
        state    = "closed" if is_done else ("active" if is_active else "planned")

        pln  = float(rng.integers(60, 150))
        hv   = float(rng.integers(50, 130))
        blk  = int(rng.integers(0, 8))
        sc   = int(rng.integers(-10, 20))
        dr   = max(0, (end_dt - TODAY).days) if is_active else 0
        if is_done:
            cmp_r = np.clip(0.6+0.25*(hv/130)-0.12*(blk/8)+rng.normal(0,0.12), 0.3, 1.05)
            cmp   = round(float(pln*cmp_r), 1)
        elif is_active:
            elapsed = (TODAY - date.fromisoformat(s_start)).days
            cmp_r   = np.clip(0.4+0.3*(elapsed/7)+rng.normal(0,0.08), 0.1, 0.85)
            cmp     = round(float(pln*cmp_r), 1)
        else:
            cmp = 0.0
        pct = round(cmp/pln*100, 1) if pln>0 else 0

        sprints.append({
            "sprint_id":     f"SPR-{i:02d}",
            "sprint_name":   f"Sprint {i}",
            "sprint_number": i,
            "state":         state,
            "start_date":    s_start,
            "end_date":      s_end,
            "planned_sp":    pln,
            "completed_sp":  cmp,
            "percent_done":  pct,
            "historical_velocity": hv,
            "blocked_stories": blk,
            "scope_change":  sc,
            "days_remaining":dr,
            "success":       int(pct >= 60 and blk < 5),
            "risk_flag":     int(pct < 40 or blk >= 5),
            "team_size":     int(rng.integers(8, 20)),
            "velocity_trend":round(float(hv/pln), 2) if pln>0 else 1.0,
        })
    return sprints

def build_issues(employees, sprints, seed=42):
    rng   = np.random.default_rng(seed)
    issues = []
    iid    = 1
    for spr in sprints:
        n_issues = int(rng.integers(60, 80))  # ~550 issues total across 8 sprints
        for _ in range(n_issues):
            emp   = employees[int(rng.integers(0, len(employees)))]
            itype = str(rng.choice(ISSUE_TYPES, p=[0.30,0.30,0.22,0.08,0.10]))
            pri   = str(rng.choice(PRIORITIES,  p=[0.08,0.28,0.44,0.20]))
            lbl   = str(rng.choice(LABELS))
            smry  = str(rng.choice(SUMMARIES))
            sp    = int(rng.choice([1,2,3,5,8,13]))
            eh    = round(float(np.clip(rng.exponential(8), 1, 60)), 1)

            # Status based on sprint state
            if spr["state"] == "closed":
                stat = str(rng.choice(STATUSES, p=[0.02,0.03,0.05,0.85,0.05]))
            elif spr["state"] == "active":
                stat = str(rng.choice(STATUSES, p=[0.15,0.40,0.18,0.22,0.05]))
            else:
                stat = "To Do"

            # Resolution time
            if stat in ["Done","In Review"]:
                ttr = round(eh * float(rng.uniform(0.6, 1.9)) + float(rng.normal(0, 1.5)), 1)
                ttr = max(0.5, ttr)
                rdate = (date.fromisoformat(spr["start_date"]) +
                         timedelta(days=int(rng.integers(1, 7)))).strftime("%Y-%m-%d")
            else:
                ttr   = eh
                rdate = None

            cdate = (date.fromisoformat(spr["start_date"]) +
                     timedelta(days=int(rng.integers(0, 5)))).strftime("%Y-%m-%d")

            issues.append({
                "issue_id":       f"AGI-{iid:04d}",
                "sprint_id":      spr["sprint_id"],
                "sprint_name":    spr["sprint_name"],
                "summary":        smry,
                "issue_type":     itype,
                "priority":       pri,
                "status":         stat,
                "assignee":       emp["name"],
                "employee_id":    emp["employee_id"],
                "team":           emp["team"],
                "story_points":   sp,
                "original_estimate_hours": eh,
                "time_spent_hours":   ttr if stat=="Done" else 0.0,
                "resolution_time_hours": ttr,
                "labels":         lbl,
                "created_date":   cdate,
                "resolved_date":  rdate,
            })
            iid += 1
    return issues

def build_ml_dataset(employees, sprints, issues, seed=42):
    """Per-row ML dataset with probabilistic labels — no label leakage."""
    rng  = np.random.default_rng(seed)
    rows = []

    for iss in issues:
        spr = next((s for s in sprints if s["sprint_id"]==iss["sprint_id"]), {})
        emp = next((e for e in employees if e["name"]==iss["assignee"]), {})

        pln  = float(spr.get("planned_sp", 80))
        hv   = float(spr.get("historical_velocity", 60))
        blk  = float(spr.get("blocked_stories", 2))
        sc   = float(spr.get("scope_change", 0))
        dr   = float(spr.get("days_remaining", 3))
        cmp  = float(spr.get("completed_sp", 50))
        pct  = float(spr.get("percent_done", 60))

        base_v = float(emp.get("base_velocity", 40))
        exp    = float(emp.get("experience_yrs", 3))
        hist_sp= base_v + float(rng.normal(0, 4))
        asgn_sp= float(iss.get("story_points", 3)) * float(rng.uniform(3, 8))
        wl     = np.clip(80 + (asgn_sp - hist_sp)*1.1 + float(rng.normal(0, 12)), 40, 200)
        hpt    = int(rng.integers(0, 7))
        co     = int(rng.integers(0, 5))
        rdr    = float(rng.integers(1, 10))

        # Probabilistic labels — correlated with features + noise
        succ_p = np.clip(1/(1+math.exp(-(pct-62)/8.0 + blk*0.28 - dr*0.03 + max(0,sc)*0.05))
                         + float(rng.normal(0, 0.04)), 0.03, 0.97)
        ol_p   = np.clip(1/(1+math.exp(-(wl-112)/8.0 + (hpt-3)*0.14))
                         + float(rng.normal(0, 0.04)), 0.02, 0.98)
        burn_p = np.clip(1/(1+math.exp(-(co-2.2)*1.0  - (wl-118)/16.0))
                         + float(rng.normal(0, 0.04)), 0.02, 0.98)

        eh  = float(iss.get("original_estimate_hours", 8))
        sp  = float(iss.get("story_points", 3))
        pri = iss.get("priority","Medium")
        ttr = float(iss.get("resolution_time_hours", eh))

        rows.append({
            # Sprint (Obj1)
            "Planned_Story_Points_Sprint":   pln,
            "Completed_Story_Points":        cmp,
            "Percent_Done":                  pct,
            "Days_Remaining_Sprint":         dr,
            "Historical_Velocity":           hv,
            "Blocked_Stories":               blk,
            "Scope_Change":                  sc,
            "Success_Label":                 int(float(rng.random()) < succ_p),
            "Sprint_Number":                 int(spr.get("sprint_number", 1)),
            # Workload (Obj2)
            "Planned_Story_Points_Resource": round(hist_sp*0.9, 1),
            "Current_Assigned_SP":           round(asgn_sp, 1),
            "Historical_Avg_SP":             round(hist_sp, 1),
            "Remaining_Days_Resource":       rdr,
            "High_Priority_Tasks_Resource":  float(hpt),
            "Current_Workload_Percent":      round(wl, 1),
            "Expected_Overload":             int(float(rng.random()) < ol_p),
            # TTR (Obj3)
            "Issue_Type":                    iss.get("issue_type","Task"),
            "Priority":                      pri,
            "Original_Estimate_Hours":       eh,
            "Story_Points_Issue":            sp,
            "Resolution_Time_Hours":         ttr,
            # Burnout (Obj4)
            "Total_SP_This_Sprint":          pln,
            "Historical_Avg_SP_Burnout":     round(hist_sp*0.85, 1),
            "High_Priority_Tasks_Burnout":   float(hpt),
            "Consecutive_Overloads":         co,
            "Risk_Flag":                     int(float(rng.random()) < burn_p),
            # Allocation (Obj5)
            "Summary":                       iss.get("summary",""),
            "Labels":                        iss.get("labels","general"),
            "Original_Estimate_Resource":    eh,
            "Story_Points_Resource":         sp,
            "Assignee_Resource":             iss.get("assignee","Unknown"),
            "Assignee":                      iss.get("assignee","Unknown"),
            # Meta
            "Issue_ID":                      iss.get("issue_id",""),
            "Status":                        iss.get("status","To Do"),
            "Team":                          iss.get("team",""),
            "Experience_Yrs":                exp,
        })
    return rows

def build_employee_stats(employees, issues, seed=42):
    rng = np.random.default_rng(seed)
    stats = []
    for emp in employees:
        emp_issues = [i for i in issues if i["employee_id"]==emp["employee_id"]]
        done  = [i for i in emp_issues if i["status"]=="Done"]
        blkd  = [i for i in emp_issues if i["status"]=="Blocked"]
        total_sp = sum(i["story_points"] for i in emp_issues)
        done_sp  = sum(i["story_points"] for i in done)
        avg_ttr  = round(np.mean([i["resolution_time_hours"] for i in done]) if done else 0, 1)
        wl   = round(float(rng.uniform(70, 160)), 1)
        co   = int(rng.integers(0, 5))
        hpt  = int(rng.integers(0, 6))
        burn = round(min(100, co*15 + max(0, wl-100)*0.5), 1)
        hlth = round(max(0, 100 - co*15 - max(0, wl-100)*0.5), 1)
        stats.append({
            **emp,
            "total_issues":      len(emp_issues),
            "completed_issues":  len(done),
            "blocked_issues":    len(blkd),
            "total_sp":          total_sp,
            "completed_sp":      done_sp,
            "completion_rate":   round(done_sp/max(total_sp,1)*100, 1),
            "avg_ttr_hours":     avg_ttr,
            "workload_pct":      wl,
            "consecutive_ol":    co,
            "high_priority":     hpt,
            "burnout_score":     burn,
            "health_score":      hlth,
            "overload":          int(wl > 110),
            "risk_flag":         int(co >= 2 or wl > 130),
        })
    return stats

# ── Build everything ──────────────────────────────────────────────────────
_employees  = build_employees(42)
_sprints    = build_sprints(42)
_issues     = build_issues(_employees, _sprints, 42)
_ml_dataset = build_ml_dataset(_employees, _sprints, _issues, 42)
_emp_stats  = build_employee_stats(_employees, _issues, 42)
_comments: dict = {}

_dist = {
    "success_pct":  round(sum(r["Success_Label"] for r in _ml_dataset)/len(_ml_dataset)*100, 1),
    "overload_pct": round(sum(r["Expected_Overload"] for r in _ml_dataset)/len(_ml_dataset)*100, 1),
    "burnout_pct":  round(sum(r["Risk_Flag"] for r in _ml_dataset)/len(_ml_dataset)*100, 1),
}

# ══════════════════════════════════════════════════════════════════════════
# HTML DATA WEBSITE — Full editorial design
# ══════════════════════════════════════════════════════════════════════════
def make_html():
    n_done   = sum(1 for i in _issues if i["status"]=="Done")
    n_risk   = sum(1 for s in _sprints if s["risk_flag"])
    n_burn   = sum(1 for e in _emp_stats if e["risk_flag"])
    avg_vel  = round(sum(s["historical_velocity"] for s in _sprints)/len(_sprints), 1)
    date_str = f"{START_DATE.strftime('%b %d')} – {TODAY.strftime('%b %d, %Y')}"

    sprint_rows = ""
    for s in _sprints:
        state_cls = "active" if s["state"]=="active" else ("planned" if s["state"]=="planned" else "closed")
        state_lbl = s["state"].upper()
        bar = min(100, s["percent_done"])
        sprint_rows += f"""
        <tr>
          <td><b>{s["sprint_name"]}</b></td>
          <td>{s["start_date"]} → {s["end_date"]}</td>
          <td><span class="tag tag-{state_cls}">{state_lbl}</span></td>
          <td>{s["planned_sp"]:.0f} SP</td>
          <td>{s["completed_sp"]:.0f} SP</td>
          <td>
            <div class="mini-bar"><div class="mini-fill" style="width:{bar:.0f}%"></div></div>
            <span class="pct-lbl">{s["percent_done"]:.1f}%</span>
          </td>
          <td>{"✓" if s["success"] else "⚠"}</td>
        </tr>"""

    team_rows = ""
    for e in sorted(_emp_stats, key=lambda x: -x["burnout_score"])[:12]:
        risk_cls = "tag-closed" if e["risk_flag"] else "tag-active"
        team_rows += f"""
        <tr>
          <td><b>{e["name"]}</b></td>
          <td>{e["role"]}</td>
          <td>{e["team"]}</td>
          <td>{e["total_issues"]}</td>
          <td>{e["completion_rate"]:.0f}%</td>
          <td>{e["workload_pct"]:.0f}%</td>
          <td>{e["burnout_score"]:.0f}</td>
          <td><span class="tag {risk_cls}">{"AT RISK" if e["risk_flag"] else "OK"}</span></td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Agile Intelligence — Data Platform</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500;600&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{{
  --cream:#f5f0e8; --cream2:#ede7d9; --cream3:#e0d8c8; --cream4:#d4c9b5;
  --forest:#1a3c2e; --forest2:#234d3b; --forest3:#2d6147; --forest4:#3a7a5a;
  --moss:#4a7c59; --sage:#6fa37d; --fern:#9bc4a7; --mist:#c5deca;
  --ink:#14201b; --bark:#4a3b2e; --stone:#7a7060; --sand:#b8ae9e;
  --amber:#e8b84b; --gold:#c4973a; --rust:#b85c38;
  --white:#fff; --red:#c0392b; --green:#27ae60;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
html{{scroll-behavior:smooth;font-size:15px;}}
body{{background:var(--cream);color:var(--ink);font-family:'DM Sans',sans-serif;
  line-height:1.6;}}

/* ── TOPBAR ── */
.topbar{{background:var(--cream2);border-bottom:1px solid var(--cream3);
  padding:0.5rem 3rem;display:flex;align-items:center;justify-content:space-between;
  font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--stone);}}
.topbar-left{{display:flex;gap:2rem;}}
.topbar-right{{display:flex;gap:1.5rem;align-items:center;}}
.topbar-right a{{color:var(--moss);text-decoration:none;}}
.topbar-right a:hover{{color:var(--forest);}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:0.3}}}}
.live{{color:var(--moss);display:flex;align-items:center;gap:5px;}}
.live::before{{content:'';display:inline-block;width:6px;height:6px;border-radius:50%;
  background:var(--moss);animation:pulse 2s infinite;}}

/* ── NAV ── */
nav{{background:var(--forest);position:sticky;top:0;z-index:100;
  border-bottom:2px solid var(--forest2);}}
.nav-inner{{max-width:1280px;margin:0 auto;padding:0 3rem;height:62px;
  display:flex;align-items:center;justify-content:space-between;}}
.nav-brand{{font-family:'DM Serif Display',serif;font-size:1.2rem;
  color:var(--cream);display:flex;align-items:center;gap:12px;letter-spacing:0.01em;}}
.brand-gem{{width:11px;height:11px;background:var(--amber);
  border-radius:2px;transform:rotate(45deg);
  box-shadow:0 0 10px #e8b84b60;}}
.nav-links{{display:flex;gap:0;}}
.nav-link{{font-family:'DM Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--mist);padding:0 1.2rem;height:62px;
  display:flex;align-items:center;text-decoration:none;border-bottom:2px solid transparent;
  transition:all 0.15s;}}
.nav-link:hover,.nav-link.active{{color:var(--cream);border-bottom-color:var(--amber);}}
.nav-actions{{display:flex;gap:8px;align-items:center;}}
.btn{{font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:600;
  letter-spacing:0.09em;text-transform:uppercase;padding:7px 16px;
  border-radius:2px;text-decoration:none;cursor:pointer;border:none;
  transition:all 0.15s;}}
.btn-ghost{{background:transparent;color:var(--fern);border:1px solid #ffffff20;}}
.btn-ghost:hover{{background:#ffffff10;}}
.btn-amber{{background:var(--amber);color:var(--forest);}}
.btn-amber:hover{{background:#f5c84e;}}

/* ── HERO ── */
.hero{{background:var(--forest);overflow:hidden;position:relative;}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 60% 80% at 70% 50%,#2d614720,transparent);}}
.hero-grid{{position:absolute;inset:0;
  background-image:linear-gradient(var(--forest3)10 1px,transparent 1px),
                   linear-gradient(90deg,var(--forest3)10 1px,transparent 1px);
  background-size:40px 40px;opacity:0.3;}}
.hero-inner{{max-width:1280px;margin:0 auto;padding:5rem 3rem 4rem;
  display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:center;
  position:relative;z-index:1;}}
.hero-eyebrow{{font-family:'DM Mono',monospace;font-size:0.65rem;
  letter-spacing:0.18em;text-transform:uppercase;color:var(--fern);
  margin-bottom:1.2rem;display:flex;align-items:center;gap:10px;}}
.hero-eyebrow::before{{content:'';width:28px;height:1px;background:var(--sage);}}
.hero-title{{font-family:'DM Serif Display',serif;font-size:3.2rem;
  color:var(--cream);line-height:1.06;letter-spacing:-0.02em;margin-bottom:1.2rem;}}
.hero-title em{{color:var(--amber);font-style:italic;}}
.hero-body{{font-size:0.95rem;color:var(--fern);line-height:1.8;
  margin-bottom:2.2rem;font-weight:300;}}
.hero-btns{{display:flex;gap:12px;flex-wrap:wrap;}}
.hero-right{{display:flex;flex-direction:column;gap:12px;}}
.stat-pill{{background:#ffffff08;border:1px solid #ffffff15;border-radius:3px;
  padding:1rem 1.3rem;display:flex;align-items:center;justify-content:space-between;}}
.sp-val{{font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--amber);}}
.sp-lbl{{font-family:'DM Mono',monospace;font-size:0.6rem;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--sage);margin-top:2px;}}
.sp-sub{{font-size:0.72rem;color:var(--stone);margin-top:1px;}}

/* ── SECTION ── */
.section{{max-width:1280px;margin:0 auto;padding:4rem 3rem;}}
.section-header{{display:flex;align-items:baseline;justify-content:space-between;
  margin-bottom:2rem;padding-bottom:0.8rem;border-bottom:1px solid var(--cream3);}}
.section-title{{font-family:'DM Serif Display',serif;font-size:1.6rem;
  color:var(--forest);letter-spacing:-0.01em;}}
.section-meta{{font-family:'DM Mono',monospace;font-size:0.6rem;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--stone);}}

/* ── KPI GRID ── */
.kpi-grid{{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;
  background:var(--cream3);border:1px solid var(--cream3);margin-bottom:3rem;}}
.kpi-cell{{background:var(--cream);padding:1.5rem 1.4rem;}}
.kpi-val{{font-family:'DM Serif Display',serif;font-size:2rem;
  color:var(--forest);line-height:1;}}
.kpi-lbl{{font-family:'DM Mono',monospace;font-size:0.58rem;
  letter-spacing:0.12em;text-transform:uppercase;color:var(--stone);margin-top:6px;}}
.kpi-delta{{font-size:0.72rem;margin-top:3px;}}
.pos{{color:var(--moss);}} .neg{{color:var(--rust);}} .neu{{color:var(--stone);}}

/* ── CARDS ── */
.card-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5rem;margin-bottom:3rem;}}
.card{{background:var(--cream);border:1px solid var(--cream3);border-radius:2px;
  padding:1.5rem 1.6rem;position:relative;overflow:hidden;transition:all 0.15s;}}
.card:hover{{border-color:var(--cream4);box-shadow:0 4px 20px #1a3c2e08;}}
.card-accent{{position:absolute;top:0;left:0;right:0;height:3px;}}
.card-eyebrow{{font-family:'DM Mono',monospace;font-size:0.58rem;
  letter-spacing:0.14em;text-transform:uppercase;color:var(--stone);margin-bottom:6px;}}
.card-title{{font-family:'DM Serif Display',serif;font-size:1.1rem;
  color:var(--forest);margin-bottom:8px;}}
.card-body{{font-size:0.82rem;color:var(--stone);line-height:1.6;}}
.card-link{{display:inline-flex;align-items:center;gap:5px;
  font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--moss);text-decoration:none;
  margin-top:1rem;}}
.card-link:hover{{color:var(--forest);}}

/* ── ENDPOINT TABLE ── */
.ep-table-wrap{{background:var(--cream);border:1px solid var(--cream3);border-radius:2px;overflow:hidden;}}
.ep-group{{border-bottom:1px solid var(--cream3);}}
.ep-group-hdr{{background:var(--cream2);padding:0.6rem 1.5rem;
  font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--forest);font-weight:600;
  display:flex;align-items:center;gap:10px;}}
.ep-row{{display:grid;grid-template-columns:80px 280px 1fr 160px;
  align-items:center;padding:0.85rem 1.5rem;border-top:1px solid var(--cream3);
  transition:background 0.1s;}}
.ep-row:hover{{background:var(--cream2);}}
.method{{font-family:'DM Mono',monospace;font-size:0.62rem;font-weight:600;
  letter-spacing:0.07em;padding:3px 9px;border-radius:2px;text-align:center;}}
.get {{background:#1a3c2e18;color:var(--forest3);}}
.post{{background:#c4973a22;color:var(--bark);}}
.ep-path{{font-family:'DM Mono',monospace;font-size:0.8rem;color:var(--forest);font-weight:500;}}
.ep-desc{{font-size:0.78rem;color:var(--stone);}}
.ep-tag{{font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.07em;
  color:var(--stone);text-align:right;}}

/* ── DATA TABLES ── */
.data-section{{background:var(--cream);border:1px solid var(--cream3);
  border-radius:2px;overflow:hidden;margin-bottom:2rem;}}
.data-hdr{{background:var(--forest);color:var(--cream);padding:1rem 1.5rem;
  display:flex;align-items:center;justify-content:space-between;}}
.data-hdr-title{{font-family:'DM Serif Display',serif;font-size:0.95rem;}}
.data-hdr-meta{{font-family:'DM Mono',monospace;font-size:0.6rem;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--fern);}}
table{{width:100%;border-collapse:collapse;}}
th{{font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.1em;
  text-transform:uppercase;color:var(--stone);padding:0.7rem 1rem;
  border-bottom:1px solid var(--cream3);text-align:left;background:var(--cream2);}}
td{{padding:0.65rem 1rem;border-bottom:1px solid var(--cream2);
  font-size:0.8rem;color:var(--ink);}}
tr:last-child td{{border-bottom:none;}}
tr:hover td{{background:var(--cream2);}}
.tag{{font-family:'DM Mono',monospace;font-size:0.55rem;font-weight:600;
  letter-spacing:0.07em;text-transform:uppercase;padding:2px 8px;border-radius:2px;}}
.tag-active {{background:#4a7c5922;color:var(--forest3);}}
.tag-closed {{background:#c0392b18;color:var(--red);}}
.tag-planned{{background:#c4973a22;color:var(--bark);}}
.mini-bar{{background:var(--cream3);border-radius:1px;height:4px;width:80px;display:inline-block;vertical-align:middle;margin-right:6px;}}
.mini-fill{{height:4px;border-radius:1px;background:var(--moss);}}
.pct-lbl{{font-family:'DM Mono',monospace;font-size:0.68rem;color:var(--forest);vertical-align:middle;}}

/* ── DIST BOX ── */
.dist-box{{background:var(--forest);padding:2.5rem 3rem;
  display:grid;grid-template-columns:repeat(4,1fr);gap:3rem;
  margin:3rem 0;border:1px solid var(--forest2);border-radius:2px;}}
.dist-item .dv{{font-family:'DM Serif Display',serif;font-size:2.4rem;
  color:var(--amber);line-height:1;}}
.dist-item .dl{{font-family:'DM Mono',monospace;font-size:0.6rem;
  letter-spacing:0.1em;text-transform:uppercase;color:var(--fern);margin-top:5px;}}
.dist-item .ds{{font-size:0.78rem;color:var(--sage);margin-top:3px;}}

/* ── FOOTER ── */
footer{{background:var(--forest);border-top:1px solid var(--forest2);}}
.footer-inner{{max-width:1280px;margin:0 auto;padding:3rem;
  display:grid;grid-template-columns:2fr 1fr 1fr 1fr;gap:3rem;}}
.footer-brand{{font-family:'DM Serif Display',serif;font-size:1rem;
  color:var(--cream);margin-bottom:0.8rem;}}
.footer-body{{font-size:0.8rem;color:var(--sage);line-height:1.6;}}
.footer-col-title{{font-family:'DM Mono',monospace;font-size:0.6rem;
  letter-spacing:0.12em;text-transform:uppercase;color:var(--fern);
  margin-bottom:0.8rem;}}
.footer-links{{list-style:none;display:flex;flex-direction:column;gap:0.5rem;}}
.footer-links a{{font-size:0.8rem;color:var(--sage);text-decoration:none;}}
.footer-links a:hover{{color:var(--cream);}}
.footer-bottom{{max-width:1280px;margin:0 auto;padding:1.2rem 3rem;
  border-top:1px solid var(--forest2);display:flex;justify-content:space-between;
  font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.07em;
  text-transform:uppercase;color:var(--stone);}}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="topbar">
  <div class="topbar-left">
    <span>Data Period: {date_str}</span>
    <span>50 Employees · 8 Sprints · {len(_issues)} Issues</span>
  </div>
  <div class="topbar-right">
    <span class="live">Live</span>
    <a href="/docs">API Docs</a>
    <a href="/redoc">ReDoc</a>
    <a href="/api/health">Health</a>
  </div>
</div>

<!-- NAV -->
<nav>
  <div class="nav-inner">
    <div class="nav-brand"><div class="brand-gem"></div>Agile Intelligence</div>
    <div class="nav-links">
      <a class="nav-link active" href="#">Overview</a>
      <a class="nav-link" href="#endpoints">API Endpoints</a>
      <a class="nav-link" href="#sprints">Sprints</a>
      <a class="nav-link" href="#team">Team</a>
      <a class="nav-link" href="#schema">Schema</a>
    </div>
    <div class="nav-actions">
      <a class="btn btn-ghost" href="/docs">Swagger ↗</a>
      <a class="btn btn-amber" href="/api/dataset/ml">ML Dataset</a>
    </div>
  </div>
</nav>

<!-- HERO -->
<div class="hero">
  <div class="hero-grid"></div>
  <div class="hero-inner">
    <div>
      <div class="hero-eyebrow">Agile Intelligence · Data Platform v3.0</div>
      <h1 class="hero-title">Two months of<br><em>agile intelligence,</em><br>served as an API.</h1>
      <p class="hero-body">
        A production-grade REST API delivering {len(_issues)} real agile issues across
        8 sprints, 50 team members, and {len(_ml_dataset)} ML-ready records —
        engineered with probabilistic labels and per-row noise for realistic model training.
      </p>
      <div class="hero-btns">
        <a class="btn btn-amber" href="/api/dataset/ml">Get ML Dataset</a>
        <a class="btn btn-ghost" href="/docs">Explore API</a>
        <a class="btn btn-ghost" href="/api/analytics/summary">Analytics</a>
      </div>
    </div>
    <div class="hero-right">
      <div class="stat-pill">
        <div><div class="sp-val">{len(_issues)}</div><div class="sp-lbl">Total Issues</div></div>
        <div style="text-align:right"><div class="sp-val" style="font-size:1.2rem;color:var(--fern)">{date_str}</div><div class="sp-sub">Data window</div></div>
      </div>
      <div class="stat-pill">
        <div><div class="sp-val">50</div><div class="sp-lbl">Team Members</div></div>
        <div style="text-align:right"><div class="sp-val" style="font-size:1.2rem;color:var(--fern)">{len(set(e["team"] for e in _employees))}</div><div class="sp-sub">Teams</div></div>
      </div>
      <div class="stat-pill">
        <div><div class="sp-val">{len(_ml_dataset)}</div><div class="sp-lbl">ML Records</div></div>
        <div style="text-align:right"><div class="sp-val" style="font-size:1.2rem;color:var(--fern)">32</div><div class="sp-sub">Feature columns</div></div>
      </div>
      <div class="stat-pill">
        <div><div class="sp-val">{_dist["success_pct"]}%</div><div class="sp-lbl">Sprint Success Rate</div></div>
        <div style="text-align:right"><div class="sp-val" style="font-size:1.2rem;color:var(--fern)">{_dist["burnout_pct"]}%</div><div class="sp-sub">Burnout risk rate</div></div>
      </div>
    </div>
  </div>
</div>

<!-- KPI STRIP -->
<div class="section" style="padding-top:3rem;padding-bottom:0;">
  <div class="kpi-grid">
    <div class="kpi-cell"><div class="kpi-val">{len(_issues)}</div><div class="kpi-lbl">Total Issues</div><div class="kpi-delta neu">↗ 2 months</div></div>
    <div class="kpi-cell"><div class="kpi-val">{n_done}</div><div class="kpi-lbl">Completed</div><div class="kpi-delta pos">↗ {round(n_done/len(_issues)*100)}% done rate</div></div>
    <div class="kpi-cell"><div class="kpi-val">50</div><div class="kpi-lbl">Engineers</div><div class="kpi-delta neu">↗ {len(set(e["team"] for e in _employees))} teams</div></div>
    <div class="kpi-cell"><div class="kpi-val">8</div><div class="kpi-lbl">Sprints</div><div class="kpi-delta neu">↗ {date_str}</div></div>
    <div class="kpi-cell"><div class="kpi-val">{n_risk}</div><div class="kpi-lbl">At-Risk Sprints</div><div class="kpi-delta {"neg" if n_risk>2 else "pos"}">{"↘ needs attention" if n_risk>2 else "↗ under control"}</div></div>
    <div class="kpi-cell"><div class="kpi-val">{avg_vel:.0f}</div><div class="kpi-lbl">Avg Velocity (SP)</div><div class="kpi-delta neu">↗ per sprint</div></div>
  </div>
</div>

<!-- WHAT'S AVAILABLE -->
<div class="section" id="endpoints">
  <div class="section-header">
    <div class="section-title">Available Data & Endpoints</div>
    <div class="section-meta">16 endpoints · REST/JSON</div>
  </div>

  <div class="card-grid">
    <div class="card">
      <div class="card-accent" style="background:var(--forest);"></div>
      <div class="card-eyebrow">ML Dataset</div>
      <div class="card-title">Complete Training Dataset</div>
      <div class="card-body">{len(_ml_dataset)} records, 32 columns, probabilistic labels for 5 ML objectives. Per-row noise prevents label leakage.</div>
      <a class="card-link" href="/api/dataset/ml">GET /api/dataset/ml →</a>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--moss);"></div>
      <div class="card-eyebrow">Sprint Data</div>
      <div class="card-title">8 Sprints Over 2 Months</div>
      <div class="card-body">Velocity, completion %, blocked stories, scope change, risk flags. Active, closed and planned states.</div>
      <a class="card-link" href="/api/sprints">GET /api/sprints →</a>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--amber);"></div>
      <div class="card-eyebrow">Team Analytics</div>
      <div class="card-title">50 Engineers, {len(set(e["team"] for e in _employees))} Teams</div>
      <div class="card-body">Burnout scores, health index, workload %, completion rates, consecutive overloads per employee.</div>
      <a class="card-link" href="/api/team">GET /api/team →</a>
    </div>
  </div>

  <!-- Endpoint table -->
  <div class="ep-table-wrap">
    <div class="ep-group">
      <div class="ep-group-hdr">◆ ML & Analytics</div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/dataset/ml</span><span class="ep-desc">Full ML-ready dataset — {len(_ml_dataset)} rows, 32 columns, probabilistic labels</span><span class="ep-tag">ML Dataset</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/analytics/summary</span><span class="ep-desc">Project health score, KPIs, risk counts, active sprint</span><span class="ep-tag">Analytics</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/analytics/velocity</span><span class="ep-desc">Sprint velocity trend over 8 sprints — completed vs planned</span><span class="ep-tag">Analytics</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/analytics/burnout</span><span class="ep-desc">Burnout risk ranked for all 50 team members</span><span class="ep-tag">Analytics</span></div>
    </div>
    <div class="ep-group">
      <div class="ep-group-hdr">◈ Sprint & Issue Data</div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/sprints</span><span class="ep-desc">All 8 sprints with velocity, completion %, risk flags</span><span class="ep-tag">Sprints</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/sprints/{{id}}</span><span class="ep-desc">Single sprint detail with issues breakdown</span><span class="ep-tag">Sprints</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/issues</span><span class="ep-desc">All {len(_issues)} issues — filter by assignee, priority, status, type</span><span class="ep-tag">Issues</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/issues/{{id}}</span><span class="ep-desc">Single issue detail with full time tracking</span><span class="ep-tag">Issues</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/search</span><span class="ep-desc">Search by keyword, assignee, priority or status</span><span class="ep-tag">Search</span></div>
    </div>
    <div class="ep-group">
      <div class="ep-group-hdr">◉ Team Data</div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/team</span><span class="ep-desc">All 50 team members with burnout scores and health index</span><span class="ep-tag">Team</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/team/{{name}}</span><span class="ep-desc">Single member detail — issues, stats, workload</span><span class="ep-tag">Team</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/employees</span><span class="ep-desc">Employee directory — 50 people, roles, teams, joined dates</span><span class="ep-tag">Team</span></div>
    </div>
    <div class="ep-group">
      <div class="ep-group-hdr">✦ Write & Admin</div>
      <div class="ep-row"><span class="method post">POST</span><span class="ep-path">/api/issues</span><span class="ep-desc">Create new issue with type, priority, story points</span><span class="ep-tag">Write</span></div>
      <div class="ep-row"><span class="method post">POST</span><span class="ep-path">/api/issues/{{id}}/comment</span><span class="ep-desc">Add comment to existing issue</span><span class="ep-tag">Write</span></div>
      <div class="ep-row"><span class="method post">POST</span><span class="ep-path">/api/issues/{{id}}/transition</span><span class="ep-desc">Transition status — To Do → In Progress → Done</span><span class="ep-tag">Write</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/refresh</span><span class="ep-desc">Regenerate all data with a new random seed</span><span class="ep-tag">Admin</span></div>
      <div class="ep-row"><span class="method get">GET</span><span class="ep-path">/api/health</span><span class="ep-desc">Health check — status, version, record counts</span><span class="ep-tag">Admin</span></div>
    </div>
  </div>
</div>

<!-- DIST BOX -->
<div class="section" style="padding-top:0;">
  <div class="dist-box">
    <div class="dist-item"><div class="dv">{_dist["success_pct"]}%</div><div class="dl">Sprint Success Rate</div><div class="ds">Probabilistic · per-row noise</div></div>
    <div class="dist-item"><div class="dv">{_dist["overload_pct"]}%</div><div class="dl">Overload Rate</div><div class="ds">Correlated with workload %</div></div>
    <div class="dist-item"><div class="dv">{_dist["burnout_pct"]}%</div><div class="dl">Burnout Risk Rate</div><div class="ds">Based on consecutive overloads</div></div>
    <div class="dist-item"><div class="dv">{len(_ml_dataset)}</div><div class="dl">ML Training Rows</div><div class="ds">32 feature columns</div></div>
  </div>
</div>

<!-- SPRINT TABLE -->
<div class="section" id="sprints" style="padding-top:0;">
  <div class="section-header">
    <div class="section-title">Sprint Summary — 8 Sprints</div>
    <div class="section-meta">{date_str}</div>
  </div>
  <div class="data-section">
    <div class="data-hdr">
      <div class="data-hdr-title">Sprint Progress</div>
      <div class="data-hdr-meta">Real dates · Velocity-correlated completion</div>
    </div>
    <table><thead><tr>
      <th>Sprint</th><th>Dates</th><th>State</th>
      <th>Planned</th><th>Completed</th><th>Progress</th><th>Status</th>
    </tr></thead><tbody>{sprint_rows}</tbody></table>
  </div>
</div>

<!-- TEAM TABLE -->
<div class="section" id="team" style="padding-top:0;">
  <div class="section-header">
    <div class="section-title">Team Health — Top Risk Members</div>
    <div class="section-meta">50 employees · Burnout index</div>
  </div>
  <div class="data-section">
    <div class="data-hdr">
      <div class="data-hdr-title">Employee Risk Matrix</div>
      <div class="data-hdr-meta">Sorted by burnout score descending · showing top 12</div>
    </div>
    <table><thead><tr>
      <th>Name</th><th>Role</th><th>Team</th>
      <th>Issues</th><th>Done Rate</th><th>Workload</th>
      <th>Burnout</th><th>Status</th>
    </tr></thead><tbody>{team_rows}</tbody></table>
  </div>
  <p style="font-size:0.75rem;color:var(--stone);margin-top:0.7rem;font-family:'DM Mono',monospace;">
    → <a href="/api/team" style="color:var(--moss);">GET /api/team</a> for all 50 members
  </p>
</div>

<!-- SCHEMA -->
<div class="section" id="schema" style="padding-top:0;padding-bottom:5rem;">
  <div class="section-header">
    <div class="section-title">ML Dataset Schema</div>
    <div class="section-meta">32 columns · 5 ML objectives</div>
  </div>
  <div class="card-grid">
    <div class="card">
      <div class="card-accent" style="background:var(--forest);"></div>
      <div class="card-eyebrow">Objective 1 — Sprint</div>
      <div class="card-title">Completion Prediction</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Planned_Story_Points_Sprint<br>Completed_Story_Points<br>
        Percent_Done · Days_Remaining_Sprint<br>Historical_Velocity · Blocked_Stories<br>
        Scope_Change · <b style="color:var(--forest);">Success_Label</b>
      </div>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--moss);"></div>
      <div class="card-eyebrow">Objective 2 — Workload</div>
      <div class="card-title">Overload Projection</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Current_Assigned_SP · Historical_Avg_SP<br>
        Remaining_Days_Resource<br>High_Priority_Tasks_Resource<br>
        Current_Workload_Percent<br><b style="color:var(--forest);">Expected_Overload</b>
      </div>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--amber);"></div>
      <div class="card-eyebrow">Objective 3 — TTR</div>
      <div class="card-title">Time to Resolve</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Issue_Type · Priority<br>
        Original_Estimate_Hours<br>Story_Points_Issue<br>
        <b style="color:var(--forest);">Resolution_Time_Hours</b>
      </div>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--rust);"></div>
      <div class="card-eyebrow">Objective 4 — Burnout</div>
      <div class="card-title">Risk Detection</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Total_SP_This_Sprint<br>Historical_Avg_SP_Burnout<br>
        High_Priority_Tasks_Burnout<br>Consecutive_Overloads<br>
        <b style="color:var(--forest);">Risk_Flag</b>
      </div>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--sage);"></div>
      <div class="card-eyebrow">Objective 5 — Allocation</div>
      <div class="card-title">Resource Matching</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Summary · Labels<br>
        Original_Estimate_Resource<br>Story_Points_Resource<br>
        <b style="color:var(--forest);">Assignee_Resource</b>
      </div>
    </div>
    <div class="card">
      <div class="card-accent" style="background:var(--bark);"></div>
      <div class="card-eyebrow">Meta Columns</div>
      <div class="card-title">Context Fields</div>
      <div class="card-body" style="font-family:'DM Mono',monospace;font-size:0.72rem;line-height:2;color:var(--stone);">
        Issue_ID · Status · Team<br>
        Assignee · Sprint_Number<br>Experience_Yrs<br>
        <a href="/api/dataset/ml" style="color:var(--moss);">GET /api/dataset/ml →</a>
      </div>
    </div>
  </div>
</div>

<!-- FOOTER -->
<footer>
  <div class="footer-inner">
    <div>
      <div class="footer-brand">Agile Intelligence</div>
      <div class="footer-body">A production-grade REST API delivering realistic
      agile data for ML training. 50 engineers · 8 sprints · 2 months of data.</div>
    </div>
    <div>
      <div class="footer-col-title">API</div>
      <ul class="footer-links">
        <li><a href="/docs">Swagger Docs</a></li>
        <li><a href="/redoc">ReDoc</a></li>
        <li><a href="/api/health">Health Check</a></li>
        <li><a href="/api/dataset/ml">ML Dataset</a></li>
      </ul>
    </div>
    <div>
      <div class="footer-col-title">Data</div>
      <ul class="footer-links">
        <li><a href="/api/sprints">Sprints</a></li>
        <li><a href="/api/team">Team</a></li>
        <li><a href="/api/employees">Employees</a></li>
        <li><a href="/api/issues">Issues</a></li>
      </ul>
    </div>
    <div>
      <div class="footer-col-title">Analytics</div>
      <ul class="footer-links">
        <li><a href="/api/analytics/summary">Summary</a></li>
        <li><a href="/api/analytics/velocity">Velocity</a></li>
        <li><a href="/api/analytics/burnout">Burnout</a></li>
        <li><a href="/api/refresh">Refresh Data</a></li>
      </ul>
    </div>
  </div>
  <div class="footer-bottom">
    <span>Agile Intelligence Data Platform · v3.0.0 · FastAPI</span>
    <span>{date_str} · {len(_issues)} Issues · 50 Engineers</span>
  </div>
</footer>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse, tags=["Root"])
def root():
    return make_html()

# ══════════════════════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════
@app.get("/api/health", tags=["Admin"])
def health():
    return {"status":"ok","version":"3.0.0","timestamp":datetime.now().isoformat(),
            "data_period":{"start":START_DATE.isoformat(),"end":TODAY.isoformat(),"days":60},
            "records":{"ml_dataset":len(_ml_dataset),"issues":len(_issues),
                       "sprints":len(_sprints),"employees":len(_employees)}}

@app.get("/api/refresh", tags=["Admin"])
def refresh():
    global _employees,_sprints,_issues,_ml_dataset,_emp_stats,_dist
    seed=random.randint(1,9999)
    _employees =build_employees(seed)
    _sprints   =build_sprints(seed)
    _issues    =build_issues(_employees,_sprints,seed)
    _ml_dataset=build_ml_dataset(_employees,_sprints,_issues,seed)
    _emp_stats =build_employee_stats(_employees,_issues,seed)
    _dist={"success_pct":round(sum(r["Success_Label"] for r in _ml_dataset)/len(_ml_dataset)*100,1),
           "overload_pct":round(sum(r["Expected_Overload"] for r in _ml_dataset)/len(_ml_dataset)*100,1),
           "burnout_pct":round(sum(r["Risk_Flag"] for r in _ml_dataset)/len(_ml_dataset)*100,1)}
    return {"status":"refreshed","seed":seed,"timestamp":datetime.now().isoformat()}

@app.get("/api/dataset/ml", tags=["ML"])
def get_ml_dataset(limit:int=Query(1000,le=2000)):
    rows=_ml_dataset[:limit]
    return {"count":len(rows),"columns":list(rows[0].keys()) if rows else [],
            "source":"Agile Intelligence API v4.0.0",
            "data_period":{"start":START_DATE.isoformat(),"end":TODAY.isoformat()},
            "employees":50,"sprints":8,"fetched_at":datetime.now().isoformat(),
            "label_distribution":_dist,"records":rows}

@app.get("/api/dataset/csv", tags=["ML"])
def get_ml_csv(limit:int=Query(1000,le=2000)):
    """Download ML dataset as a CSV file."""
    import io as _io, csv as _csv
    rows=_ml_dataset[:limit]
    if not rows: raise HTTPException(404,"No data")
    buf=_io.StringIO()
    w=_csv.DictWriter(buf,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows); buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(_io.BytesIO(buf.getvalue().encode()),media_type="text/csv",
        headers={"Content-Disposition":f"attachment; filename=agile_ml_dataset.csv"})

@app.get("/api/sprints", tags=["Sprints"])
def get_sprints(state:Optional[str]=None):
    d=_sprints if not state else [s for s in _sprints if s["state"]==state]
    return {"count":len(d),"sprints":d}

@app.get("/api/sprints/{sprint_id}", tags=["Sprints"])
def get_sprint(sprint_id:str):
    s=next((x for x in _sprints if x["sprint_id"]==sprint_id),None)
    if not s: raise HTTPException(404,f"Sprint {sprint_id} not found")
    sprint_issues=[i for i in _issues if i["sprint_id"]==sprint_id]
    return {**s,"issues_count":len(sprint_issues),"issues":sprint_issues[:20]}

@app.get("/api/issues", tags=["Issues"])
def get_issues(assignee:Optional[str]=None,priority:Optional[str]=None,
               status:Optional[str]=None,issue_type:Optional[str]=None,
               sprint_id:Optional[str]=None,team:Optional[str]=None,
               limit:int=Query(200,le=1000)):
    d=_issues
    if assignee:   d=[i for i in d if i["assignee"]==assignee]
    if priority:   d=[i for i in d if i["priority"]==priority]
    if status:     d=[i for i in d if i["status"]==status]
    if issue_type: d=[i for i in d if i["issue_type"]==issue_type]
    if sprint_id:  d=[i for i in d if i["sprint_id"]==sprint_id]
    if team:       d=[i for i in d if i["team"]==team]
    return {"count":len(d[:limit]),"issues":d[:limit]}

@app.get("/api/issues/{issue_id}", tags=["Issues"])
def get_issue(issue_id:str):
    i=next((x for x in _issues if x["issue_id"]==issue_id),None)
    if not i: raise HTTPException(404,f"Issue {issue_id} not found")
    return {**i,"comments":_comments.get(issue_id,[])}

@app.get("/api/employees", tags=["Team"])
def get_employees(team:Optional[str]=None,role:Optional[str]=None):
    d=_employees
    if team: d=[e for e in d if e["team"]==team]
    if role: d=[e for e in d if e["role"]==role]
    return {"count":len(d),"employees":d}

@app.get("/api/team", tags=["Team"])
def get_team(team:Optional[str]=None):
    d=_emp_stats
    if team: d=[e for e in d if e["team"]==team]
    return {"count":len(d),"team":d}

@app.get("/api/team/{name}", tags=["Team"])
def get_member(name:str):
    m=next((x for x in _emp_stats if x["name"].lower()==name.lower()),None)
    if not m: raise HTTPException(404,f"Member {name} not found")
    emp_issues=[i for i in _issues if i["assignee"]==m["name"]]
    return {**m,"recent_issues":emp_issues[:10]}

@app.get("/api/search", tags=["Search"])
def search(q:str="",assignee:Optional[str]=None,
           priority:Optional[str]=None,status:Optional[str]=None,
           team:Optional[str]=None,limit:int=Query(50,le=300)):
    d=_issues
    if q:        d=[i for i in d if q.lower() in i["summary"].lower() or q.lower() in i["labels"]]
    if assignee: d=[i for i in d if i["assignee"]==assignee]
    if priority: d=[i for i in d if i["priority"]==priority]
    if status:   d=[i for i in d if i["status"]==status]
    if team:     d=[i for i in d if i["team"]==team]
    return {"query":q,"count":len(d[:limit]),"results":d[:limit]}

@app.get("/api/analytics/summary", tags=["Analytics"])
def summary():
    done =sum(1 for i in _issues if i["status"]=="Done")
    blk  =sum(1 for i in _issues if i["status"]=="Blocked")
    hp   =sum(1 for i in _issues if i["priority"] in ["Critical","High"])
    risk =sum(1 for s in _sprints if s["risk_flag"])
    burn =sum(1 for e in _emp_stats if e["risk_flag"])
    over =sum(1 for e in _emp_stats if e["overload"])
    av   =round(sum(s["historical_velocity"] for s in _sprints)/len(_sprints),1)
    ac   =round(sum(s["percent_done"] for s in _sprints if s["state"]!="planned")/
                max(1,sum(1 for s in _sprints if s["state"]!="planned")),1)
    h    =min(100,max(0,100-risk*8-burn*4-blk//8*2-over*3))
    return {"health_score":h,"total_issues":len(_issues),"done":done,"blocked":blk,
            "high_priority":hp,"sprints_at_risk":risk,"burnout_risk":burn,"overloaded":over,
            "avg_velocity":av,"avg_completion_pct":ac,"team_size":50,
            "data_period":{"start":START_DATE.isoformat(),"end":TODAY.isoformat()},
            "active_sprint":next((s for s in _sprints if s["state"]=="active"),None),
            "generated_at":datetime.now().isoformat()}

@app.get("/api/analytics/velocity", tags=["Analytics"])
def velocity():
    return {"sprints":[{"id":s["sprint_id"],"name":s["sprint_name"],"state":s["state"],
            "start":s["start_date"],"end":s["end_date"],
            "velocity":s["historical_velocity"],"completed":s["completed_sp"],
            "planned":s["planned_sp"],"pct":s["percent_done"],"risk":bool(s["risk_flag"])}
            for s in _sprints]}

@app.get("/api/analytics/burnout", tags=["Analytics"])
def burnout_rpt():
    return {"count":len(_emp_stats),
            "team":sorted([{"name":e["name"],"team":e["team"],"role":e["role"],
            "burnout_score":e["burnout_score"],"health_score":e["health_score"],
            "workload_pct":e["workload_pct"],"consec_ol":e["consecutive_ol"],
            "risk_flag":e["risk_flag"],"high_priority":e["high_priority"]}
            for e in _emp_stats],key=lambda x:-x["burnout_score"])}

class IssueCreate(BaseModel):
    summary:str; issue_type:str="Task"; priority:str="Medium"
    assignee:str="Unassigned"; story_points:int=3; labels:str="general"; sprint_id:str="SPR-08"

class CommentCreate(BaseModel):
    text:str; author:str="API User"

class TransitionCreate(BaseModel):
    status:str

@app.post("/api/issues",tags=["Write"],status_code=201)
def create_issue(body:IssueCreate):
    nid=f"AGI-{len(_issues)+1:04d}"; eh=round(body.story_points*float(np.random.uniform(1.5,2.5)),1)
    row={"issue_id":nid,"sprint_id":body.sprint_id,"sprint_name":"Sprint 8",
         "summary":body.summary,"issue_type":body.issue_type,"priority":body.priority,
         "status":"To Do","assignee":body.assignee,"employee_id":"EMP-000",
         "team":"Unknown","story_points":body.story_points,
         "original_estimate_hours":eh,"time_spent_hours":0.0,
         "resolution_time_hours":eh,"labels":body.labels,
         "created_date":datetime.now().strftime("%Y-%m-%d"),"resolved_date":None}
    _issues.append(row); return {"created":True,"issue_id":nid,"issue":row}

@app.post("/api/issues/{issue_id}/comment",tags=["Write"])
def add_comment(issue_id:str,body:CommentCreate):
    if issue_id not in _comments: _comments[issue_id]=[]
    c={"id":len(_comments[issue_id])+1,"author":body.author,
       "text":body.text,"at":datetime.now().isoformat()}
    _comments[issue_id].append(c); return {"added":True,"comment":c}

@app.post("/api/issues/{issue_id}/transition",tags=["Write"])
def transition(issue_id:str,body:TransitionCreate):
    valid=["To Do","In Progress","In Review","Done","Blocked"]
    if body.status not in valid: raise HTTPException(400,f"Status must be one of {valid}")
    row=next((r for r in _issues if r.get("issue_id")==issue_id),None)
    if not row: raise HTTPException(404,f"{issue_id} not found")
    old=row["status"]; row["status"]=body.status
    return {"transitioned":True,"issue_id":issue_id,"from":old,"to":body.status}