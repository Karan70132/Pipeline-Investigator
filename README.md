# Silent Data Error Investigator

A successful SQL query can still report the wrong revenue. This project trains a random forest to distinguish a real sales increase from double-counted orders and missing records.

## Run in your browser

1. Open Train_in_Colab.ipynb via File > Upload notebook in Google Colab.
2. Connect to a CPU runtime. Choose Runtime > Run all.
3. When prompted, select Pipeline-Investigator-GitHub.zip. The notebook installs dependencies, trains a fresh model, prints four predictions and downloads the new results.
4. Open metrics.json in the downloaded results to inspect class-level scores and the confusion matrix.

## How it works

The simulator creates orders, order items and payments in SQLite. An erroneous join repeats order totals for multi-item orders. A missing partition drops one order-ID bucket. Healthy examples represent ordinary demand and genuine growth.

Each simulated day produces four features: relative revenue change, relative order-count change, dashboard rows per unique order, and reported revenue divided by payments. A random forest with 80 trees and maximum depth 6 predicts the scenario from those four numbers.

## Training and results

The training set has 800 examples. Evaluation uses 200 examples with distinct seeds and 50 per class. The submitted Colab run scored 200/200 with macro F1 of 1.00. Both splits use the same rules. Refunds, partial payments and unseen incidents require separate evaluation. The current implementation is a supervised ML baseline. An LLM investigation layer is future work.

## Repository files

- investigator.py generates data, trains the classifier and runs inference.
- Train_in_Colab.ipynb runs the pipeline in a browser.
- requirements.txt lists training dependencies.
- saved_run/ contains the original Colab model and metrics. Fresh training writes to outputs/ separately. Only load trusted joblib files.
- README.pdf contains the printable project and publishing guide.
