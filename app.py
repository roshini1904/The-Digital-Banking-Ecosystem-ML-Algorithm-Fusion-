
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, flash, session)
from functools import wraps
import json, datetime, os, sqlite3, hashlib, secrets, math, random

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
DB_FILE = "securebank.db"


# ═══════════════════════════════════════════════════════════
#  PURE PYTHON UTILITIES
# ═══════════════════════════════════════════════════════════

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def minmax_normalize(X):
    """Column-wise min-max normalization. Returns (X_norm, mins, maxs)."""
    cols = len(X[0])
    mins = [min(row[c] for row in X) for c in range(cols)]
    maxs = [max(row[c] for row in X) for c in range(cols)]
    X_norm = []
    for row in X:
        norm = []
        for c in range(cols):
            denom = maxs[c] - mins[c] if maxs[c] != mins[c] else 1
            norm.append((row[c] - mins[c]) / denom)
        X_norm.append(norm)
    return X_norm, mins, maxs

def normalize_row(row, mins, maxs):
    norm = []
    for c, v in enumerate(row):
        denom = maxs[c] - mins[c] if maxs[c] != mins[c] else 1
        norm.append((v - mins[c]) / denom)
    return norm

def train_test_split(X, y, test_ratio=0.25, seed=42):
    random.seed(seed)
    data = list(zip(X, y))
    random.shuffle(data)
    split = int(len(data) * (1 - test_ratio))
    train = data[:split]
    test  = data[split:]
    X_tr, y_tr = zip(*train)
    X_te, y_te = zip(*test)
    return list(X_tr), list(y_tr), list(X_te), list(y_te)

def confusion_matrix(y_true, y_pred, labels=(0, 1)):
    """Returns dict with TP, TN, FP, FN."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}

def eval_metrics(y_true, y_pred):
    """Accuracy, Precision, Recall, F1 from predictions."""
    cm   = confusion_matrix(y_true, y_pred)
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    total    = tp + tn + fp + fn
    accuracy  = round((tp + tn) / total * 100, 2)       if total            else 0
    precision = round(tp / (tp + fp) * 100, 2)           if (tp + fp)        else 0
    recall    = round(tp / (tp + fn) * 100, 2)           if (tp + fn)        else 0
    f1        = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0
    return {
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1_score":  f1,
        "confusion_matrix": cm,
        "total_samples": total
    }


# ═══════════════════════════════════════════════════════════
#  1. KNN LOAN APPROVAL MODEL
# ═══════════════════════════════════════════════════════════

class KNNLoanModel:
    """
    K-Nearest Neighbors Classifier — Loan Approval.

    Features : income (USD), credit_score (300-850), loan_amount (USD)
    Labels   : 1 = Approved, 0 = Rejected
    Distance : Euclidean on min-max normalized features
    """
    def __init__(self, k=3):
        self.k       = k
        self.X_train = []
        self.y_train = []
        self.mins    = []
        self.maxs    = []
        self.eval    = {}

    def train(self, X, y):
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.25)
        self.X_train, self.mins, self.maxs = minmax_normalize(X_tr)
        self.y_train = y_tr
        # Evaluate on test set
        preds = [self._raw_predict(row) for row in X_te]
        self.eval = eval_metrics(y_te, preds)
        self.eval["test_samples"] = len(y_te)

    def _raw_predict(self, raw_row):
        point = normalize_row(raw_row, self.mins, self.maxs)
        dists = []
        for i, tr in enumerate(self.X_train):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, tr)))
            dists.append((d, self.y_train[i]))
        dists.sort(key=lambda x: x[0])
        votes = [n[1] for n in dists[:self.k]]
        return 1 if votes.count(1) > self.k / 2 else 0

    def predict(self, data_dict):
        income = float(data_dict.get("income", 50000))
        credit = float(data_dict.get("credit_score", 650))
        amount = float(data_dict.get("amount", 10000))
        raw    = [income, credit, amount]
        pred   = self._raw_predict(raw)

        point = normalize_row(raw, self.mins, self.maxs)
        dists = []
        for i, tr in enumerate(self.X_train):
            d = math.sqrt(sum((a - b) ** 2 for a, b in zip(point, tr)))
            dists.append((d, self.y_train[i]))
        dists.sort(key=lambda x: x[0])
        votes        = [n[1] for n in dists[:self.k]]
        approved_cnt = votes.count(1)
        confidence   = int(max(approved_cnt, self.k - approved_cnt) / self.k * 100)

        return {
            "status":             "Approved" if pred == 1 else "Rejected",
            "approved":           pred == 1,
            "confidence":         f"{confidence}%",
            "neighbors_checked":  self.k,
            "votes_for_approval": approved_cnt,
        }

    def metrics(self):
        return {
            "algorithm":        "K-Nearest Neighbors (KNN)",
            "k_value":          self.k,
            "distance_metric":  "Euclidean Distance",
            "normalization":    "Min-Max Scaling",
            "training_samples": len(self.y_train),
            "features":         ["Annual Income", "Credit Score", "Loan Amount"],
            "evaluation":       self.eval,
            "status":           "Active"
        }


# ═══════════════════════════════════════════════════════════
#  2. REAL RANDOM FOREST — FRAUD DETECTION
# ═══════════════════════════════════════════════════════════

class DecisionTree:
    """
    A single CART-style Decision Tree (binary splits, Gini impurity).
    Supports bootstrap sampling and random feature subsets for RF.
    """
    def __init__(self, max_depth=4, min_samples=2, n_features=None, seed=0):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.n_features  = n_features
        self.seed        = seed
        self.tree        = None

    def _gini(self, groups, classes):
        total = sum(len(g) for g in groups)
        score = 0.0
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            p_sum = 0.0
            for cls in classes:
                p = sum(1 for row in group if row[-1] == cls) / size
                p_sum += p * p
            score += (1 - p_sum) * (size / total)
        return score

    def _split(self, index, value, dataset):
        left  = [row for row in dataset if row[index] < value]
        right = [row for row in dataset if row[index] >= value]
        return left, right

    def _best_split(self, dataset, rng):
        classes   = list(set(row[-1] for row in dataset))
        n_cols    = len(dataset[0]) - 1
        feat_idxs = list(range(n_cols))
        if self.n_features and self.n_features < n_cols:
            feat_idxs = rng.sample(feat_idxs, self.n_features)

        best_idx, best_val, best_score, best_groups = None, None, float('inf'), None
        for idx in feat_idxs:
            for row in dataset:
                groups = self._split(idx, row[idx], dataset)
                score  = self._gini(groups, classes)
                if score < best_score:
                    best_idx, best_val, best_score, best_groups = idx, row[idx], score, groups
        return {"index": best_idx, "value": best_val, "groups": best_groups}

    def _leaf(self, group):
        labels = [row[-1] for row in group]
        return max(set(labels), key=labels.count)

    def _build(self, node, depth, rng):
        left, right = node["groups"]
        del node["groups"]
        if not left or not right:
            node["left"] = node["right"] = self._leaf(left + right)
            return
        if depth >= self.max_depth:
            node["left"]  = self._leaf(left)
            node["right"] = self._leaf(right)
            return
        if len(left) <= self.min_samples:
            node["left"] = self._leaf(left)
        else:
            node["left"] = self._best_split(left, rng)
            self._build(node["left"], depth + 1, rng)
        if len(right) <= self.min_samples:
            node["right"] = self._leaf(right)
        else:
            node["right"] = self._best_split(right, rng)
            self._build(node["right"], depth + 1, rng)

    def fit(self, X, y):
        rng     = random.Random(self.seed)
        dataset = [list(x) + [label] for x, label in zip(X, y)]
        n       = len(dataset)
        # Bootstrap sample
        sample  = [rng.choice(dataset) for _ in range(n)]
        self.tree = self._best_split(sample, rng)
        self._build(self.tree, 1, rng)

    def _predict_row(self, node, row):
        if isinstance(node, dict):
            branch = node["left"] if row[node["index"]] < node["value"] else node["right"]
            return self._predict_row(branch, row)
        return node

    def predict(self, X):
        return [self._predict_row(self.tree, row) for row in X]


class RandomForestFraudModel:
    """
    Real Random Forest Classifier for Fraud Detection.

    Features : amount (USD), location_mismatch (0/1), late_night (0/1)
    Labels   : 1 = Fraud, 0 = Legitimate
    Method   : Bootstrap + random feature subsets + majority vote
    """
    def __init__(self, n_trees=15, max_depth=4, n_features=2):
        self.n_trees    = n_trees
        self.max_depth  = max_depth
        self.n_features = n_features
        self.trees      = []
        self.mins       = []
        self.maxs       = []
        self.eval       = {}
        self.feat_importance = {"amount": 0, "location_mismatch": 0, "late_night": 0}

    def train(self, X, y):
        X_tr, y_tr, X_te, y_te = train_test_split(X, y, test_ratio=0.25, seed=7)
        X_norm, self.mins, self.maxs = minmax_normalize(X_tr)
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                                n_features=self.n_features, seed=i * 13)
            tree.fit(X_norm, y_tr)
            self.trees.append(tree)
        # Evaluate
        X_te_norm = [normalize_row(r, self.mins, self.maxs) for r in X_te]
        preds = self._ensemble_predict(X_te_norm)
        self.eval = eval_metrics(y_te, preds)
        self.eval["test_samples"] = len(y_te)
        # Rough feature importance by counting split features
        self._compute_feature_importance()

    def _count_splits(self, node, counts):
        if isinstance(node, dict):
            idx = node.get("index")
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1
            self._count_splits(node.get("left"), counts)
            self._count_splits(node.get("right"), counts)

    def _compute_feature_importance(self):
        counts = {}
        for tree in self.trees:
            self._count_splits(tree.tree, counts)
        total = sum(counts.values()) or 1
        keys  = ["amount", "location_mismatch", "late_night"]
        self.feat_importance = {keys[i]: round(counts.get(i, 0) / total * 100, 1)
                                for i in range(3)}

    def _ensemble_predict(self, X_norm):
        all_preds = [tree.predict(X_norm) for tree in self.trees]
        result = []
        for i in range(len(X_norm)):
            votes = [all_preds[t][i] for t in range(len(self.trees))]
            result.append(1 if votes.count(1) > len(self.trees) / 2 else 0)
        return result

    def predict(self, data_dict):
        amount    = float(data_dict.get("amount", 0))
        loc_mm    = int(data_dict.get("location_mismatch", 0))
        late_n    = int(data_dict.get("late_night", 0))
        raw       = [[amount, loc_mm, late_n]]
        norm      = [normalize_row(raw[0], self.mins, self.maxs)]
        all_votes = [tree.predict(norm)[0] for tree in self.trees]
        fraud_v   = all_votes.count(1)
        is_fraud  = fraud_v > len(self.trees) / 2
        risk_pct  = int(fraud_v / len(self.trees) * 100)
        return {
            "status":      "Fraudulent" if is_fraud else "Clear",
            "is_fraud":    is_fraud,
            "risk_score":  f"{risk_pct}%",
            "fraud_votes": fraud_v,
            "total_trees": len(self.trees),
        }

    def metrics(self):
        return {
            "algorithm":          "Random Forest (Real CART Trees)",
            "total_trees":        self.n_trees,
            "max_depth":          self.max_depth,
            "features_per_split": self.n_features,
            "voting_strategy":    "Majority Vote",
            "feature_importance": self.feat_importance,
            "evaluation":         self.eval,
            "status":             "Active"
        }


# ═══════════════════════════════════════════════════════════
#  3. Q-LEARNING RL AGENT — CREDIT ADJUSTMENT
# ═══════════════════════════════════════════════════════════

class QLearningCreditAgent:
    """
    Q-Learning RL Agent for credit limit decisions.

    States  : 8 discrete states (balance + payment history)
    Actions : 0=Decrease(-$500)  1=Hold($0)  2=Increase(+$1500)
    Update  : Bellman equation  Q(s,a) += α[r + γ·maxQ(s') - Q(s,a)]
    """
    def __init__(self, n_states=8, n_actions=3):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.gamma     = 0.9
        self.alpha     = 0.1
        self.q_table   = [[0.0] * n_actions for _ in range(n_states)]
        self.rewards_log = []   # tracks avg reward per episode for chart

    def train(self, episodes=500):
        rng = random.Random(42)
        for ep in range(episodes):
            state      = rng.randint(0, self.n_states - 1)
            ep_rewards = []
            for _ in range(10):
                action = rng.randint(0, self.n_actions - 1)
                if state >= 5:
                    reward = [-1.0, 0.3, 1.5][action]
                elif state >= 3:
                    reward = [0.2, 0.8, 0.1][action]
                else:
                    reward = [1.2, 0.4, -0.5][action]
                next_state = min(state + (1 if action == 2 else 0), self.n_states - 1)
                best_next  = max(self.q_table[next_state])
                self.q_table[state][action] += self.alpha * (
                    reward + self.gamma * best_next - self.q_table[state][action]
                )
                ep_rewards.append(reward)
                state = next_state
            if ep % 50 == 0:
                self.rewards_log.append(round(sum(ep_rewards) / len(ep_rewards), 3))

    def decide(self, data_dict):
        balance = float(data_dict.get("balance", 5000))
        missed  = int(data_dict.get("missed_payments", 0))
        state   = min(max(int(balance / 3000) - missed, 0), self.n_states - 1)
        row     = self.q_table[state]
        action  = row.index(max(row))

        labels     = {0: "Decrease Limit", 1: "Hold Steady", 2: "Increase Limit"}
        changes    = {0: -500, 1: 0, 2: 1500}
        rationale  = {
            0: "High missed payments or low balance — reducing credit exposure.",
            1: "Stable profile — maintaining current credit limit.",
            2: "Strong balance and payment history — rewarding with higher credit.",
        }
        return {
            "state_index":  state,
            "action":       labels[action],
            "amount_change": changes[action],
            "q_values":     [round(v, 4) for v in row],
            "rationale":    rationale[action],
        }

    def get_qtable(self):
        return [[round(v, 4) for v in row] for row in self.q_table]

    def metrics(self):
        return {
            "algorithm":      "Q-Learning (Reinforcement Learning)",
            "gamma_discount": self.gamma,
            "learning_rate":  self.alpha,
            "state_space":    self.n_states,
            "action_space":   self.n_actions,
            "episodes_trained": 500,
            "rewards_per_checkpoint": self.rewards_log,
            "mode":   "Exploitation (training complete)",
            "status": "Active"
        }


# ═══════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role     TEXT NOT NULL DEFAULT 'customer',
                name     TEXT,
                email    TEXT,
                created  TEXT
            );
            CREATE TABLE IF NOT EXISTS decision_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                model_type TEXT,
                username   TEXT,
                payload    TEXT
            );

            CREATE TABLE IF NOT EXISTS credit_requests (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                username        TEXT NOT NULL,
                current_limit   REAL DEFAULT 50000,
                requested_limit REAL NOT NULL,
                reason          TEXT,
                status          TEXT DEFAULT 'Pending',
                ai_recommendation TEXT,
                ai_confidence   TEXT,
                admin_note      TEXT,
                submitted_at    TEXT,
                reviewed_at     TEXT,
                reviewed_by     TEXT
            );
        """)

def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def seed_default_users():
    with get_db() as conn:
        if conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO users (username,password,role,name,email,created) VALUES (?,?,?,?,?,?)",
                ("admin", hash_password("Admin@123"), "admin",
                 "Bank Administrator", "admin@securebank.com", now))
            conn.execute(
                "INSERT INTO users (username,password,role,name,email,created) VALUES (?,?,?,?,?,?)",
                ("customer", hash_password("Customer@123"), "customer",
                 "Demo Customer", "customer@securebank.com", now))
            print("✅  Seeded: admin/Admin@123  and  customer/Customer@123")

if os.path.exists(DB_FILE):
    try:
        with sqlite3.connect(DB_FILE) as c:
            c.execute("SELECT username FROM users LIMIT 1")
    except sqlite3.OperationalError:
        os.remove(DB_FILE)

init_db()
seed_default_users()


# ═══════════════════════════════════════════════════════════
#  TRAINING DATA  (50 synthetic samples each)
# ═══════════════════════════════════════════════════════════

# --- KNN: [income, credit_score, loan_amount], label (1=Approved) ---
LOAN_X = [
    [80000,750,20000],[30000,580,40000],[120000,800,10000],[45000,620,15000],
    [25000,500,5000],[95000,710,50000],[60000,690,12000],[35000,540,30000],
    [150000,820,8000],[70000,740,18000],[55000,670,22000],[40000,600,35000],
    [110000,790,15000],[28000,510,8000],[85000,760,25000],[50000,640,10000],
    [130000,810,30000],[32000,560,20000],[75000,720,18000],[42000,610,28000],
    [90000,755,12000],[22000,480,6000],[105000,780,20000],[48000,630,16000],
    [65000,700,14000],[38000,575,25000],[115000,795,22000],[27000,520,9000],
    [82000,748,19000],[53000,660,11000],[140000,815,35000],[36000,555,18000],
    [72000,730,16000],[44000,615,27000],[98000,765,13000],[26000,495,7000],
    [88000,758,21000],[51000,645,12000],[125000,805,28000],[33000,550,22000],
    [68000,710,15000],[41000,605,30000],[108000,785,17000],[29000,525,10000],
    [78000,745,20000],[46000,625,14000],[135000,812,25000],[31000,545,19000],
    [62000,695,13000],[57000,675,16000]
]
LOAN_Y = [1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,1,1,0,1,0,
          1,0,1,1,1,0,1,0,1,1,1,0,1,0,1,0,1,1,1,0,
          1,0,1,0,1,1,1,0,1,1]

# --- RF: [amount, location_mismatch, late_night], label (1=Fraud) ---
FRAUD_X = [
    [250,0,0],[6500,1,1],[120,0,0],[8200,1,0],[300,0,1],
    [9500,1,1],[75,0,0],[4500,0,1],[11000,1,1],[200,0,0],
    [7800,1,0],[150,0,0],[5500,1,1],[400,0,0],[12000,1,1],
    [100,0,0],[3200,0,0],[8800,1,1],[50,0,0],[6200,1,0],
    [900,0,1],[7100,1,1],[180,0,0],[5800,1,0],[350,0,0],
    [10500,1,1],[220,0,0],[4200,0,1],[9800,1,1],[130,0,0],
    [6800,1,0],[280,0,0],[11500,1,1],[160,0,0],[5100,1,0],
    [420,0,1],[8400,1,1],[95,0,0],[7500,1,0],[310,0,0],
    [12500,1,1],[240,0,0],[4800,0,1],[9200,1,1],[110,0,0],
    [6100,1,0],[380,0,0],[10200,1,1],[175,0,0],[5400,1,0]
]
FRAUD_Y = [0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,
           0,1,0,1,0,1,0,0,1,0,1,0,1,0,1,0,1,0,1,0,
           1,0,0,1,0,1,0,1,0,1]

print("⏳  Training ML models...")
knn_model = KNNLoanModel(k=3)
knn_model.train(LOAN_X, LOAN_Y)

rf_model = RandomForestFraudModel(n_trees=15, max_depth=4, n_features=2)
rf_model.train(FRAUD_X, FRAUD_Y)

rl_agent = QLearningCreditAgent()
rl_agent.train(episodes=500)
print("✅  All ML engines trained and evaluated!")


# ═══════════════════════════════════════════════════════════
#  AUTH DECORATORS
# ═══════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            flash("Access denied — admin clearance required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrapper

def log_decision(model_type, result_dict):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO decision_history (timestamp,model_type,username,payload) VALUES (?,?,?,?)",
            (result_dict["timestamp"], model_type,
             session.get("username", "unknown"), json.dumps(result_dict))
        )


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def root():
    return redirect(url_for("dashboard") if "user_id" in session else url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password=?",
                (username, hash_password(password))
            ).fetchone()
        if user:
            session.clear()
            session["user_id"]  = user["id"]
            session["username"] = user["username"]
            session["role"]     = user["role"]
            session["name"]     = user["name"]
            return redirect(url_for("dashboard"))
        flash("Account not found. Please register to create a new account.", "danger")
        return redirect(url_for("register"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username  = request.form.get("username", "").strip().lower()
        full_name = request.form.get("full_name", "").strip()
        email     = request.form.get("email", "").strip().lower()
        password  = request.form.get("password", "")
        confirm   = request.form.get("confirm_password", "")
        if not username or not full_name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")
        if len(username) < 3:
            flash("Username must be at least 3 characters.", "danger")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")
        if password != confirm:
            flash("Passwords do not match.", "danger")
            return render_template("register.html")
        if username == "admin":
            flash("That username is reserved.", "danger")
            return render_template("register.html")
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users (username,password,role,name,email,created) VALUES (?,?,?,?,?,?)",
                    (username, hash_password(password), "customer", full_name, email, now)
                )
            flash(f"Account created! Welcome, {full_name}. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already taken — please choose another.", "danger")
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    if session.get("role") == "admin":
        return render_template("admin_dashboard.html",
                               name=session["name"], username=session["username"])
    return render_template("customer_dashboard.html",
                           name=session["name"], username=session["username"])


# ═══════════════════════════════════════════════════════════
#  ML API ENDPOINTS
# ═══════════════════════════════════════════════════════════

@app.route("/api/loan", methods=["POST"])
@login_required
def api_loan():
    data   = request.get_json(force=True)
    result = knn_model.predict(data)
    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["type"]      = "KNN Loan Approval"
    result["input"]     = data
    log_decision("KNN Loan", result)
    return jsonify(result)

@app.route("/api/fraud", methods=["POST"])
@login_required
def api_fraud():
    data   = request.get_json(force=True)
    result = rf_model.predict(data)
    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["type"]      = "RF Fraud Detection"
    result["input"]     = data
    log_decision("RF Fraud", result)
    return jsonify(result)

@app.route("/api/credit", methods=["POST"])
@login_required
def api_credit():
    data   = request.get_json(force=True)
    result = rl_agent.decide(data)
    result["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result["type"]      = "RL Credit Adjustment"
    result["input"]     = data
    log_decision("RL Credit", result)
    return jsonify(result)

@app.route("/api/history")
@login_required
def api_history():
    with get_db() as conn:
        if session.get("role") == "admin":
            rows = conn.execute(
                "SELECT payload FROM decision_history ORDER BY id DESC LIMIT 100"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT payload FROM decision_history WHERE username=? ORDER BY id DESC LIMIT 50",
                (session["username"],)
            ).fetchall()
    return jsonify([json.loads(r["payload"]) for r in rows])

@app.route("/api/metrics")
@admin_required
def api_metrics():
    return jsonify({
        "knn":    knn_model.metrics(),
        "rf":     rf_model.metrics(),
        "rl":     rl_agent.metrics(),
        "qtable": rl_agent.get_qtable()
    })

@app.route("/api/analytics")
@admin_required
def api_analytics():
    with get_db() as conn:
        users = conn.execute(
            "SELECT username, role, name, email, created FROM users ORDER BY id"
        ).fetchall()
        total_decisions = conn.execute("SELECT COUNT(*) FROM decision_history").fetchone()[0]
        model_counts    = conn.execute(
            "SELECT model_type, COUNT(*) as cnt FROM decision_history GROUP BY model_type"
        ).fetchall()
        loan_stats = conn.execute(
            "SELECT payload FROM decision_history WHERE model_type='KNN Loan'"
        ).fetchall()
        fraud_stats = conn.execute(
            "SELECT payload FROM decision_history WHERE model_type='RF Fraud'"
        ).fetchall()

    loan_approved  = sum(1 for r in loan_stats  if json.loads(r["payload"]).get("approved"))
    loan_rejected  = len(loan_stats) - loan_approved
    fraud_detected = sum(1 for r in fraud_stats if json.loads(r["payload"]).get("is_fraud"))
    fraud_clear    = len(fraud_stats) - fraud_detected

    return jsonify({
        "users":            [dict(u) for u in users],
        "total_decisions":  total_decisions,
        "model_usage":      {r["model_type"]: r["cnt"] for r in model_counts},
        "loan_stats":       {"approved": loan_approved, "rejected": loan_rejected},
        "fraud_stats":      {"detected": fraud_detected, "clear": fraud_clear},
    })

# NEW: evaluation endpoint — returns accuracy, precision, recall, F1, confusion matrix
@app.route("/api/evaluation")
@admin_required
def api_evaluation():
    knn_e = knn_model.metrics()["evaluation"]
    rf_e  = rf_model.metrics()["evaluation"]
    return jsonify({
        "knn": {
            "model":      "KNN Loan Approval",
            "accuracy":   knn_e.get("accuracy",  0),
            "precision":  knn_e.get("precision", 0),
            "recall":     knn_e.get("recall",    0),
            "f1_score":   knn_e.get("f1_score",  0),
            "confusion_matrix": knn_e.get("confusion_matrix", {}),
            "test_samples": knn_e.get("test_samples", 0),
        },
        "rf": {
            "model":      "Random Forest Fraud Detection",
            "accuracy":   rf_e.get("accuracy",  0),
            "precision":  rf_e.get("precision", 0),
            "recall":     rf_e.get("recall",    0),
            "f1_score":   rf_e.get("f1_score",  0),
            "confusion_matrix": rf_e.get("confusion_matrix", {}),
            "test_samples": rf_e.get("test_samples", 0),
        },
        "rl_rewards": rl_agent.metrics()["rewards_per_checkpoint"],
        "rf_feature_importance": rf_model.feat_importance,
    })

# ═══════════════════════════════════════════════════════════
#  ADMIN MANAGEMENT API ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/api/admin/users", methods=["GET"])
@admin_required
def admin_get_users():
    with get_db() as conn:
        users = conn.execute(
            "SELECT id, username, role, name, email, created FROM users ORDER BY id"
        ).fetchall()
    return jsonify([dict(u) for u in users])

@app.route("/api/admin/users/add", methods=["POST"])
@admin_required
def admin_add_user():
    d         = request.get_json(force=True)
    username  = d.get("username", "").strip().lower()
    name      = d.get("name", "").strip()
    email     = d.get("email", "").strip().lower()
    role      = d.get("role", "customer")
    password  = d.get("password", "")
    if not username or not name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    if role not in ("admin", "customer"):
        return jsonify({"error": "Invalid role."}), 400
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users (username,password,role,name,email,created) VALUES (?,?,?,?,?,?)",
                (username, hash_password(password), role, name, email, now)
            )
        return jsonify({"success": True, "message": f"User '{username}' created successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 409

@app.route("/api/admin/users/<int:user_id>", methods=["GET"])
@admin_required
def admin_get_user(user_id):
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, username, role, name, email, created FROM users WHERE id=?", (user_id,)
        ).fetchone()
    if not user:
        return jsonify({"error": "User not found."}), 404
    return jsonify(dict(user))

@app.route("/api/admin/users/<int:user_id>/edit", methods=["POST"])
@admin_required
def admin_edit_user(user_id):
    d        = request.get_json(force=True)
    name     = d.get("name", "").strip()
    email    = d.get("email", "").strip().lower()
    role     = d.get("role", "customer")
    password = d.get("password", "").strip()
    if not name or not email:
        return jsonify({"error": "Name and email are required."}), 400
    if role not in ("admin", "customer"):
        return jsonify({"error": "Invalid role."}), 400
    # Prevent removing the only admin
    with get_db() as conn:
        target = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not target:
            return jsonify({"error": "User not found."}), 404
        if target["role"] == "admin" and role == "customer":
            admin_count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin'"
            ).fetchone()[0]
            if admin_count <= 1:
                return jsonify({"error": "Cannot demote the only admin account."}), 400
        if password:
            if len(password) < 6:
                return jsonify({"error": "New password must be at least 6 characters."}), 400
            conn.execute(
                "UPDATE users SET name=?, email=?, role=?, password=? WHERE id=?",
                (name, email, role, hash_password(password), user_id)
            )
        else:
            conn.execute(
                "UPDATE users SET name=?, email=?, role=? WHERE id=?",
                (name, email, role, user_id)
            )
    return jsonify({"success": True, "message": "User updated successfully."})

@app.route("/api/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def admin_delete_user(user_id):
    with get_db() as conn:
        target = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not target:
            return jsonify({"error": "User not found."}), 404
        if target["username"] == "admin":
            return jsonify({"error": "Cannot delete the primary admin account."}), 400
        if target["username"] == session.get("username"):
            return jsonify({"error": "Cannot delete your own account."}), 400
        if target["role"] == "admin":
            admin_count = conn.execute(
                "SELECT COUNT(*) FROM users WHERE role='admin'"
            ).fetchone()[0]
            if admin_count <= 1:
                return jsonify({"error": "Cannot delete the only admin account."}), 400
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.execute("DELETE FROM decision_history WHERE username=?", (target["username"],))
    return jsonify({"success": True, "message": f"User '{target['username']}' deleted."})

@app.route("/api/admin/users/<int:user_id>/reset-password", methods=["POST"])
@admin_required
def admin_reset_password(user_id):
    d           = request.get_json(force=True)
    new_password = d.get("password", "").strip()
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    with get_db() as conn:
        user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not user:
            return jsonify({"error": "User not found."}), 404
        conn.execute(
            "UPDATE users SET password=? WHERE id=?",
            (hash_password(new_password), user_id)
        )
    return jsonify({"success": True, "message": "Password reset successfully."})

@app.route("/api/admin/logs/clear", methods=["POST"])
@admin_required
def admin_clear_logs():
    with get_db() as conn:
        conn.execute("DELETE FROM decision_history")
    return jsonify({"success": True, "message": "All decision logs cleared."})

@app.route("/api/admin/logs/delete/<int:log_id>", methods=["POST"])
@admin_required
def admin_delete_log(log_id):
    with get_db() as conn:
        conn.execute("DELETE FROM decision_history WHERE id=?", (log_id,))
    return jsonify({"success": True, "message": "Log entry deleted."})


@app.route("/api/admin/retrain", methods=["POST"])
@admin_required
def admin_retrain():
    d        = request.get_json(force=True)
    gamma    = float(d.get("gamma",    0.9))
    alpha    = float(d.get("alpha",    0.1))
    episodes = int(d.get("episodes",   500))

    # Validate ranges
    if not (0.1 <= gamma <= 1.0):
        return jsonify({"error": "Gamma must be between 0.1 and 1.0"}), 400
    if not (0.01 <= alpha <= 0.5):
        return jsonify({"error": "Alpha must be between 0.01 and 0.5"}), 400
    if not (100 <= episodes <= 5000):
        return jsonify({"error": "Episodes must be between 100 and 5000"}), 400

    # Retrain with new hyperparameters
    global rl_agent
    rl_agent         = QLearningCreditAgent()
    rl_agent.gamma   = gamma
    rl_agent.alpha   = alpha
    rl_agent.train(episodes=episodes)

    return jsonify({
        "success":  True,
        "message":  f"Agent retrained successfully with γ={gamma}, α={alpha}, episodes={episodes}",
        "qtable":   rl_agent.get_qtable(),
        "rewards":  rl_agent.metrics()["rewards_per_checkpoint"],
        "metrics":  rl_agent.metrics(),
    })


# ═══════════════════════════════════════════════════════════
#  CREDIT REQUEST ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/api/credit/request", methods=["POST"])
@login_required
def submit_credit_request():
    if session.get("role") == "admin":
        return jsonify({"error": "Admins cannot submit credit requests."}), 403
    d = request.get_json(force=True)
    requested = float(d.get("requested_limit", 0))
    current   = float(d.get("current_limit", 50000))
    reason    = d.get("reason", "").strip()

    if requested <= current:
        return jsonify({"error": "Requested limit must be higher than current limit."}), 400
    if requested > 500000:
        return jsonify({"error": "Maximum requestable limit is ₹5,00,000."}), 400
    if not reason:
        return jsonify({"error": "Please provide a reason for your request."}), 400

    # Check for existing pending request
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM credit_requests WHERE username=? AND status='Pending'",
            (session["username"],)
        ).fetchone()
        if existing:
            return jsonify({"error": "You already have a pending request. Wait for admin review."}), 400

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            """INSERT INTO credit_requests
               (username, current_limit, requested_limit, reason, status, submitted_at)
               VALUES (?,?,?,?,?,?)""",
            (session["username"], current, requested, reason, "Pending", now)
        )
    return jsonify({"success": True, "message": "Request submitted successfully! Awaiting admin review."})


@app.route("/api/credit/my-requests")
@login_required
def my_credit_requests():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT cr.*, u.name FROM credit_requests cr
               LEFT JOIN users u ON cr.username = u.username
               WHERE cr.username=? ORDER BY cr.id DESC""",
            (session["username"],)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/credit-requests")
@admin_required
def admin_get_credit_requests():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT cr.*, u.name FROM credit_requests cr
               LEFT JOIN users u ON cr.username = u.username
               ORDER BY
                 CASE cr.status WHEN 'Pending' THEN 0 ELSE 1 END,
                 cr.id DESC"""
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/admin/credit-requests/<int:req_id>/evaluate", methods=["POST"])
@admin_required
def evaluate_credit_request(req_id):
    with get_db() as conn:
        req = conn.execute(
            "SELECT * FROM credit_requests WHERE id=?", (req_id,)
        ).fetchone()
    if not req:
        return jsonify({"error": "Request not found."}), 404

    # Get customer balance and missed payments from their history
    with get_db() as conn:
        history = conn.execute(
            """SELECT payload FROM decision_history
               WHERE username=? AND model_type='RL Credit'
               ORDER BY id DESC LIMIT 1""",
            (req["username"],)
        ).fetchone()

    # Use history data if available, else defaults
    balance  = 5000
    missed   = 0
    if history:
        try:
            p = json.loads(history["payload"])
            inp = p.get("input", {})
            balance = float(inp.get("balance", 5000))
            missed  = int(inp.get("missed_payments", 0))
        except:
            pass

    result = rl_agent.decide({"balance": balance, "missed_payments": missed})
    action = result["action"]

    # Map RL action to recommendation
    if action == "Increase Limit":
        recommendation = "Approved"
        confidence     = "High — customer profile supports increase"
    elif action == "Hold Steady":
        recommendation = "Partial"
        confidence     = "Moderate — customer profile is stable but not strong"
    else:
        recommendation = "Rejected"
        confidence     = "Low — customer profile indicates financial risk"

    return jsonify({
        "recommendation": recommendation,
        "confidence":     confidence,
        "rl_action":      action,
        "q_values":       result["q_values"],
        "state_index":    result["state_index"],
        "rationale":      result["rationale"],
        "balance_used":   balance,
        "missed_used":    missed,
    })


@app.route("/api/admin/credit-requests/<int:req_id>/decision", methods=["POST"])
@admin_required
def decide_credit_request(req_id):
    d          = request.get_json(force=True)
    decision   = d.get("decision")       # "Approved" or "Rejected"
    admin_note = d.get("admin_note", "").strip()
    ai_rec     = d.get("ai_recommendation", "")
    ai_conf    = d.get("ai_confidence", "")

    if decision not in ("Approved", "Rejected"):
        return jsonify({"error": "Decision must be Approved or Rejected."}), 400

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_db() as conn:
        req = conn.execute(
            "SELECT * FROM credit_requests WHERE id=?", (req_id,)
        ).fetchone()
        if not req:
            return jsonify({"error": "Request not found."}), 404
        if req["status"] != "Pending":
            return jsonify({"error": "Request already reviewed."}), 400

        conn.execute(
            """UPDATE credit_requests SET
               status=?, ai_recommendation=?, ai_confidence=?,
               admin_note=?, reviewed_at=?, reviewed_by=?
               WHERE id=?""",
            (decision, ai_rec, ai_conf, admin_note, now,
             session["username"], req_id)
        )

        # If approved, update user's credit limit in a log entry
        if decision == "Approved":
            payload = {
                "type":      "Credit Request Approved",
                "username":  req["username"],
                "timestamp": now,
                "old_limit": req["current_limit"],
                "new_limit": req["requested_limit"],
                "approved_by": session["username"],
            }
            conn.execute(
                "INSERT INTO decision_history (timestamp,model_type,username,payload) VALUES (?,?,?,?)",
                (now, "Credit Approval", req["username"], json.dumps(payload))
            )

    return jsonify({
        "success": True,
        "message": f"Request {decision.lower()} successfully."
    })


@app.route("/api/admin/notifications")
@admin_required
def admin_notifications():
    """Returns pending credit requests count + recent activity for popup notification."""
    with get_db() as conn:
        pending = conn.execute(
            "SELECT cr.*, u.name FROM credit_requests cr LEFT JOIN users u ON cr.username=u.username WHERE cr.status='Pending' ORDER BY cr.id DESC"
        ).fetchall()
        recent_decisions = conn.execute(
            "SELECT payload, model_type FROM decision_history ORDER BY id DESC LIMIT 5"
        ).fetchall()
    return jsonify({
        "pending_count":    len(pending),
        "pending_requests": [dict(p) for p in pending],
        "recent_activity":  [{"type": r["model_type"], "data": json.loads(r["payload"])} for r in recent_decisions],
    })

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SecureBank AI — MCA Capstone Project")
    print("  http://127.0.0.1:5000")
    print("  Logins: admin/Admin@123  |  customer/Customer@123")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000)
