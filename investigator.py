"""Train and evaluate a classifier on independently simulated SQL pipeline runs."""
import argparse
import json
import random
import sqlite3
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

CASES = ["ordinary", "growth", "join_fault", "missing_partition"]
FEATURES = ["revenue_change", "order_change", "rows_per_order", "payment_reconciliation"]


def simulate(seed, case):
    rng = random.Random(seed)
    previous = rng.randint(70, 240)
    multiplier = rng.uniform(1.22, 1.45) if case == "growth" else rng.uniform(.94, 1.06)
    count = round(previous * multiplier)
    prices = [rng.randrange(100, 1800) for _ in range(count)]
    previous_revenue = previous * sum(prices) / count * rng.uniform(.94, 1.06)
    with sqlite3.connect(":memory:") as db:
        db.executescript("""
            CREATE TABLE orders(id INTEGER PRIMARY KEY, total INTEGER);
            CREATE TABLE order_items(order_id INTEGER);
            CREATE TABLE payments(order_id INTEGER, amount INTEGER);
        """)
        db.executemany("INSERT INTO orders VALUES (?,?)", enumerate(prices))
        db.executemany("INSERT INTO payments VALUES (?,?)", enumerate(prices))
        items = [(i,) for i in range(count) for _ in range(1 + int(rng.random() < .33))]
        db.executemany("INSERT INTO order_items VALUES (?)", items)
        query = "SELECT id,total FROM orders"
        if case == "join_fault":
            query += " JOIN order_items ON id=order_id"
        elif case == "missing_partition":
            query += " WHERE id % 5 != 0"
        rows, distinct_orders, revenue = db.execute(
            f"SELECT COUNT(*), COUNT(DISTINCT id), SUM(total) FROM ({query})"
        ).fetchone()
        paid = db.execute("SELECT SUM(amount) FROM payments").fetchone()[0]
    return [revenue / previous_revenue - 1, distinct_orders / previous - 1,
            rows / distinct_orders, revenue / paid]


def dataset(start, size):
    labels = [CASES[i % len(CASES)] for i in range(size)]
    return [simulate(start + i, label) for i, label in enumerate(labels)], labels


def train():
    x_train, y_train = dataset(0, 800)
    x_test, y_test = dataset(10000, 200)
    model = RandomForestClassifier(n_estimators=80, max_depth=6, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    Path("outputs").mkdir(exist_ok=True)
    joblib.dump(model, "outputs/model.joblib")
    report = classification_report(y_test, predictions, output_dict=True)
    report["label_order"] = CASES
    report["confusion_matrix"] = confusion_matrix(y_test, predictions, labels=CASES).tolist()
    Path("outputs/metrics.json").write_text(json.dumps(report, indent=2))
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "demo"])
    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        model = joblib.load("outputs/model.joblib")
        for case in CASES:
            values = simulate(20260, case)
            print(case, "=>", model.predict([values])[0], dict(zip(FEATURES, map(lambda x: round(x, 3), values))))
