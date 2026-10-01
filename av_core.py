"""
av_core.py - Core algorithms for adaptive path planning & collision avoidance on
unstructured (Indian) roads. Pure Python (stdlib only), deterministic given a seed.

Pipeline per 0.1 s planning cycle:
  detections -> alpha-beta (Kalman-style) tracker -> multi-modal motion prediction
  -> Monte-Carlo collision probability -> risk-aware lateral-lattice planner
  -> decision FSM -> longitudinal / lateral control (rate-limited)
plus a driver-impairment monitor that tightens the planner and can trigger a
minimal-risk manoeuvre (pull over and stop).

This is a 2-D kinematic simulation, not a MATLAB/Simulink or CARLA model.
"""
import math
import random
import time

TY = {
    "auto": {"w": 1.4, "l": 2.6, "k": 0, "n": "auto-rickshaw"},
    "car": {"w": 1.8, "l": 4.2, "k": 0, "n": "car"},
    "bike": {"w": 0.8, "l": 2.0, "k": 0, "n": "two-wheeler"},
    "truck": {"w": 2.4, "l": 7.0, "k": 0, "n": "truck"},
    "ped": {"w": 0.6, "l": 0.6, "k": 1, "n": "pedestrian"},
    "cow": {"w": 0.9, "l": 1.8, "k": 2, "n": "cow"},
    "cart": {"w": 1.0, "l": 1.8, "k": 2, "n": "pushcart"},
    "rock": {"w": 1.0, "l": 1.0, "k": 2, "n": "road debris"},
}
RN = ["LOW", "MEDIUM", "HIGH"]
CAP = {"OK": 1.0, "WARN": 0.9, "CONSERVATIVE": 0.7, "TAKEOVER": 0.7, "MRM": 0.0}

SC = [
    {"n": "Village road (unmarked)", "cv": 8, "hw": 3.0,
     "init": [("rock", .5, 30, 0, 0), ("cart", -1.5, 42, 0, 1.2), ("ped", -4.4, 26, 1, 0)],
     "mix": [("ped", .05), ("cow", .03), ("rock", .04), ("bike", .05), ("cart", .04)]},
    {"n": "Urban intersection (no signal)", "cv": 7, "hw": 3.5,
     "init": [("car", -9, 26, 4, 0), ("ped", 4.8, 30, -1.1, 0)],
     "mix": [("xcar", .12), ("ped", .06), ("auto", .06), ("bike", .06)]},
    {"n": "Highway merge (slow vehicles)", "cv": 12, "hw": 3.5,
     "init": [("truck", 1.9, 34, 0, 4), ("car", 5, 22, -1.6, 9), ("car", -1.9, 45, 0, 8)],
     "mix": [("truck", .04), ("car", .08), ("auto", .03), ("bike", .04)]},
    {"n": "Dense market (mixed traffic)", "cv": 5, "hw": 3.5,
     "init": [("ped", -4, 18, 1.2, 0), ("cart", 1, 24, 0, 1.2), ("auto", -1.8, 30, 0, 2), ("bike", 2.5, 22, -1, 3)],
     "mix": [("ped", .25), ("cart", .12), ("auto", .1), ("bike", .15), ("cow", .03)]},
    {"n": "Sudden cattle crossing", "cv": 8, "hw": 3.5,
     "init": [("cow", -3, 26, .8, 0), ("cow", -4.5, 32, .8, 0), ("cow", 1, 38, 0, 0), ("car", 2, 50, 0, 5)],
     "mix": [("cow", .05), ("ped", .03), ("car", .04)]},
]

CD = [-3, -2, -1, 0, 1, 2, 3]          # lateral lattice offsets (m)
TT = [.5, 1, 1.5, 2, 2.5, 3]           # prediction horizon steps (s)
N = 48                                 # Monte-Carlo samples per object


def sm(u):
    return u * u * (3 - 2 * u)


def gauss(r):
    return math.sqrt(-2 * math.log(r.random() + 1e-9)) * math.cos(6.2832 * r.random())


def rn(r, a, b):
    return a + r.random() * (b - a)


class Obj:
    def __init__(self, id_, t, x, y, vx, vy, conf=0.9):
        self.id, self.t, self.x, self.y, self.vx, self.vy, self.conf = id_, t, x, y, vx, vy, conf
        self.lvl, self.p, self.hit = 0, 0.0, False
        self.ex = None
        self.ey, self.evx, self.evy, self.sg, self.ttc, self.P = y, 0.0, 0.0, 0.6, 99.0, []


class Sim:
    def __init__(self, si=0, mode="ours", seed=5, sev=0.0, scenario=None):
        self.sc = scenario or SC[si]
        self.mode, self.sev = mode, float(sev)
        self.rng, self.nr = random.Random(seed), random.Random(seed + 99)
        self.t = self.x = self.vl = 0.0
        self.v = float(self.sc["cv"])
        self.objs, self.uid = [], 0
        self.col, self.gap, self.dist = 0, 99.0, 0.0
        self.pt = self.a2 = 0.0
        self.an = self.msN = 0
        self.msSum = 0.0
        self.lv, self.alerts = {}, []
        self.dr, self.ds, self.tk, self.mrm, self.lastC = 0.12, "OK", 0.0, False, 0.0
        self.A = {"lvl": 0, "st": "CRUISE", "tc": 0.0, "tv": float(self.sc["cv"]), "costs": [], "why": "", "ms": 0.0}
        for t, x, y, vx, vy in self.sc["init"]:
            self.objs.append(Obj(self.next_id(), t, x, y, vx, vy))

    def next_id(self):
        self.uid += 1
        return self.uid


def gen(s, t):
    r, hw = s.rng, s.sc["hw"]
    sg = -1 if r.random() < .5 else 1
    o = Obj(s.next_id(), t, 0.0, rn(r, 52, 68), 0.0, 0.0, rn(r, .84, .98))
    if t == "xcar":
        o.t, o.x, o.y, o.vx = "car", -11 * sg, rn(r, 18, 34), 4 * sg
    elif t == "ped":
        o.x, o.y, o.vx = (hw + 1.4) * sg, rn(r, 24, 48), -sg * rn(r, .8, 1.4)
    elif TY[t]["k"] == 0 or t == "cart":
        o.x = sg * hw * .6 + rn(r, -.3, .3)
        o.vy = rn(r, 1, 1.6) if t == "cart" else rn(r, 3, 7)
        if t == "bike":
            o.vx = rn(r, -1, 1)
    else:
        o.x, o.y = rn(r, -hw + .6, hw - .6), rn(r, 36, 60)
    return o


# ---------------- driver impairment monitor ----------------
def driver_signals(sev):
    return {"perclos": .08 + .5 * sev, "reaction_time_s": .3 + .9 * sev,
            "lane_weave_m": .05 + .6 * sev, "breath_alcohol_mgl": .5 * sev}


def driver_risk(perclos, react, weave, breath):
    return min(1.0, .3 * perclos / .6 + .25 * react / 1.2 + .2 * weave / .65 + .25 * breath / .5)


def driver_state(dr):
    return "OK" if dr < .3 else "WARN" if dr < .5 else "CONSERVATIVE" if dr < .75 else "TAKEOVER"


# ---------------- perception / prediction / risk ----------------
def track(s, o, dt):
    r = s.nr
    zx, zy = o.x + gauss(r) * .25, o.y + gauss(r) * .5
    if o.ex is None:
        o.ex, o.ey, o.evx, o.evy, o.sg = zx, zy, o.vx, o.vy - s.v, .6
        return
    o.ex += o.evx * dt
    o.ey += o.evy * dt
    a, b = zx - o.ex, zy - o.ey
    o.ex += .5 * a
    o.ey += .5 * b
    o.evx += .25 * a / dt
    o.evy += .25 * b / dt
    o.sg = max(.25, o.sg * .9)


def cloud(s, o):
    r = s.nr
    d = 1.0 if TY[o.t]["k"] == 1 else 0.0 if o.t == "rock" else .5
    P = []
    for _ in range(N):
        u = r.random()
        m = 0 if u < .7 else 1 if u < .85 else -1       # continue / veer right / veer left
        P.append((o.ex + gauss(r) * o.sg, o.ey, o.evx + m * d + gauss(r) * .2, o.evy + gauss(r) * .4))
    o.P = P


def pcol(o, pf, mg):
    T = TY[o.t]
    hw, ah, bh = T["w"] / 2 + .9 * mg, T["l"] / 2 + 2 * mg, -4.2 - T["l"] / 2
    n = 0
    for p in o.P:
        for t in TT:
            x, y = p[0] + p[2] * t, p[1] + p[3] * t
            if bh < y < ah and abs(x - pf(t)) < hw:
                n += 1
                break
    return n / N


def why(o, mode):
    prob = f"collision prob {int(o.p * 100)}% (Monte-Carlo {N} samples)" if mode == "ours" else "brake rule"
    ttc = "-" if o.ttc > 50 else f"{o.ttc:.1f} s"
    return f"{TY[o.t]['n']} {o.ey:.0f} m ahead, {prob}, TTC {ttc}, track sigma {o.sg:.1f} m"


def plan(s, dt, do_track=True):
    t0 = time.perf_counter()
    hw, dr = s.sc["hw"], s.dr
    mg, hiT, mdT = 1 + .5 * dr, .2 * (1 - .5 * dr), .06 * (1 - .5 * dr)
    if do_track:
        for o in s.objs:
            track(s, o, dt)
    vis = [o for o in s.objs if -4 < o.ey < 60]
    if s.mode == "ours":
        for o in vis:
            cloud(s, o)
    worst = None
    for o in vis:
        T = TY[o.t]
        o.ttc = max(o.ey - T["l"] / 2 - 2.2, 0) / -o.evy if o.evy < -.2 else 99.0
        if s.mode == "ours":
            o.p = pcol(o, lambda t: s.x, mg)
            o.lvl = 2 if o.p >= hiT else 1 if o.p >= mdT else 0
        else:
            o.p = 0.0
            o.lvl = 2 if abs(o.ex - s.x) < T["w"] / 2 + 1.2 and o.ttc < 3 else 0
        if o.ey < 6 and abs(o.ex - s.x) < 2.2:
            o.lvl = max(o.lvl, 1)
        if o.lvl > s.lv.get(o.id, 0):
            s.alerts.insert(0, {"t": round(s.t, 1), "level": RN[o.lvl], "msg": why(o, s.mode)})
        s.lv[o.id] = o.lvl
        if worst is None or o.lvl > worst.lvl or (o.lvl == worst.lvl and o.p > worst.p):
            worst = o
    lvl = worst.lvl if worst else 0
    tv, tc, st, costs = s.sc["cv"] * CAP[s.ds], 0.0, "CRUISE", []
    if s.mode == "ours":
        for c in [c for c in CD if abs(c) <= hw - .8]:
            pf = (lambda c: lambda t: s.x + (c - s.x) * sm(min(t / 2.5, 1)))(c)
            q = 1.0
            for o in vis:
                q *= 1 - pcol(o, pf, mg)
            risk = 1 - q
            cost = 10 * risk + .5 * abs(c - s.x) + .15 * abs(c - s.lastC) + .1 * abs(c) \
                + (.5 * abs(c - (hw - .9)) if s.mrm else 0)
            costs.append({"c": c, "risk": risk, "cost": cost})
        b = min(costs, key=lambda k: k["cost"])
        tc = s.lastC = b["c"]
        if lvl > 0:
            if b["risk"] < hiT:
                st, tv = "EVADE", tv * (.7 if lvl == 2 else .85)
            elif lvl == 2:
                st, tv = "EMERGENCY BRAKE", 0.0
            else:
                st, tv = "YIELD", tv * .5
    elif lvl == 2:
        st, tv = "BRAKE", 0.0
    if s.mrm and st != "EMERGENCY BRAKE":
        st = "MRM STOPPED" if s.v < .1 else "MRM PULL-OVER"
    return {"lvl": lvl, "costs": costs, "tc": tc, "tv": tv, "st": st, "worst": worst,
            "why": why(worst, s.mode) if lvl and worst else "No object on a collision course.",
            "ms": (time.perf_counter() - t0) * 1000}


def step(s, dt):
    s.t += dt
    hw = s.sc["hw"]
    s.dr += (min(1.0, driver_risk(*[driver_signals(s.sev)[k] for k in
             ("perclos", "reaction_time_s", "lane_weave_m", "breath_alcohol_mgl")])) - s.dr) * min(1, dt * 1.5)
    d = driver_state(s.dr)
    s.tk = s.tk + dt if d == "TAKEOVER" else 0.0
    if s.tk > 4:
        s.mrm = True
    if s.mrm and s.dr < .3:
        s.mrm = False
    nd = "MRM" if s.mrm else d
    if nd != s.ds:
        s.ds = nd
        s.alerts.insert(0, {"t": round(s.t, 1), "level": "HIGH" if nd in ("TAKEOVER", "MRM") else "MEDIUM" if nd != "OK" else "LOW",
                            "msg": "Driver unresponsive/impaired: minimal-risk manoeuvre, hazards ON, pulling over, emergency contact notified"
                            if nd == "MRM" else f"Driver state -> {nd}"})
    for o in s.objs:
        o.x += o.vx * dt
        o.y += (o.vy - s.v) * dt
        T = TY[o.t]
        if not o.hit and abs(o.x - s.x) < T["w"] / 2 + .9 and -4.2 - T["l"] / 2 < o.y < T["l"] / 2 + .3:
            o.hit = True
            s.col += 1
            s.alerts.insert(0, {"t": round(s.t, 1), "level": "HIGH", "msg": "COLLISION with " + T["n"]})
        if o.y > 0:
            g = math.hypot(max(abs(o.x - s.x) - T["w"] / 2 - .9, 0), max(o.y - T["l"] / 2, 0))
            s.gap = min(s.gap, g)
    s.objs = [o for o in s.objs if -8 < o.y < 80 and abs(o.x) < 14]
    if len(s.objs) < 9:
        for m, rt in s.sc["mix"]:
            if s.rng.random() < rt * dt:
                s.objs.append(gen(s, m))
    del s.alerts[7:]
    s.pt += dt
    if s.pt >= .1:
        s.A = plan(s, s.pt)
        s.pt = 0.0
        s.msSum += s.A["ms"]
        s.msN += 1
    A = s.A
    s.v += max(-(3 if s.mrm else 7) * dt, min(2 * dt, A["tv"] - s.v))
    if s.v < .05 and s.mrm:
        s.v = 0.0
    ov = s.vl
    s.vl += max(-2.5 * dt, min(2.5 * dt, (A["tc"] - s.x) - s.vl))
    s.x = max(-hw + .5, min(hw - .5, s.x + s.vl * dt))
    s.a2 += ((s.vl - ov) / dt) ** 2
    s.an += 1
    s.dist += s.v * dt


def metrics(s):
    need = .6 * s.sc["cv"] * s.t
    return {"collisions": s.col, "min_gap_m": None if s.gap > 90 else round(s.gap, 2),
            "avg_replan_ms": round(s.msSum / s.msN, 3) if s.msN else None,
            "rms_lateral_accel_mps2": round(math.sqrt(s.a2 / s.an), 3) if s.an else 0.0,
            "distance_m": round(s.dist, 1), "completed": s.col == 0 and s.dist >= need,
            "driver_state": s.ds, "final_speed_kmh": round(s.v * 3.6, 1)}


def simulate(si, mode="ours", seed=5, seconds=25.0, sev=0.0, trace=False):
    s = Sim(si, mode, seed, sev)
    tr = []
    for k in range(int(seconds / .05)):
        step(s, .05)
        if trace and k % 10 == 0:
            tr.append({"t": round(s.t, 2), "x": round(s.x, 2), "speed_kmh": round(s.v * 3.6, 1),
                       "state": s.A["st"], "risk": RN[s.A["lvl"]], "driver": s.ds})
    out = {"scenario": s.sc["n"], "planner": mode, "seconds": seconds, "metrics": metrics(s), "alerts": s.alerts}
    if trace:
        out["trace"] = tr
    return out


def benchmark(seconds=25.0):
    rows = []
    for i, sc in enumerate(SC):
        for m in ("ours", "brake"):
            r = simulate(i, m, 11 + i, seconds)
            rows.append({"scenario": sc["n"], "planner": m, **r["metrics"]})
    return rows


def assess(objects, ego_speed=9.0, ego_x=0.0, hw=3.5, sev=0.0, seed=1):
    """Single-shot risk assessment for externally supplied detections (treated as tracked estimates)."""
    s = Sim(0, "ours", seed, sev, scenario={"n": "api", "cv": ego_speed, "hw": hw, "init": [], "mix": []})
    s.x, s.v = ego_x, ego_speed
    sg = driver_signals(sev)
    s.dr = driver_risk(sg["perclos"], sg["reaction_time_s"], sg["lane_weave_m"], sg["breath_alcohol_mgl"])
    s.ds = driver_state(s.dr)
    for d in objects:
        o = Obj(s.next_id(), d["type"], d["x"], d["y"], d.get("vx", 0.0), d.get("vy", 0.0), d.get("conf", .9))
        o.ex, o.ey, o.evx, o.evy, o.sg = o.x, o.y, o.vx, o.vy - ego_speed, .3
        s.objs.append(o)
    A = plan(s, .1, do_track=False)
    return {"overall_risk": RN[A["lvl"]], "decision": A["st"], "target_lateral_offset_m": A["tc"],
            "target_speed_kmh": round(A["tv"] * 3.6, 1), "reason": A["why"], "driver_state": s.ds,
            "driver_risk": round(s.dr, 3),
            "objects": [{"id": o.id, "type": o.t, "collision_probability": round(o.p, 3), "risk": RN[o.lvl],
                         "ttc_s": None if o.ttc > 50 else round(o.ttc, 2)} for o in s.objs],
            "candidate_paths": [{"offset_m": c["c"], "collision_risk": round(c["risk"], 3), "cost": round(c["cost"], 3)}
                                for c in A["costs"]],
            "replan_latency_ms": round(A["ms"], 3)}
