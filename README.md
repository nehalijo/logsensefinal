# LogSense — Log Anomaly Detection System

## Project Description
LogSense is an intelligent log analysis system that automatically detects anomalies in system log files using machine learning. Instead of manually scanning thousands of log entries, LogSense uses a deep learning autoencoder to learn what normal system behaviour looks like and automatically flags anything unusual — helping system administrators identify failures, security threats, and unexpected behaviour early.

---

## Problem Statement
Modern computer systems generate millions of log entries every day. Manually monitoring these logs is impossible at scale. Critical issues like system failures, security breaches, and software bugs often leave traces in log files long before they cause serious damage — but only if someone is watching. LogSense automates this process entirely.

---

## How It Works
Raw log files are fed into the system and parsed into structured data. Related log events are grouped into sequences per block ID, representing the workflow of individual system operations. A machine learning model is then trained on normal sequences to learn what healthy system behaviour looks like. When the model encounters a sequence it cannot reconstruct accurately, it flags it as an anomaly.

---

## Models Used

**Autoencoder (Main Model)**
A neural network that compresses and reconstructs log sequences. Trained exclusively on normal sequences so it cannot reconstruct anomalous ones — high reconstruction error signals an anomaly. This is the core detection engine of LogSense.

**Isolation Forest (Comparison Model)**
A traditional machine learning approach that isolates anomalies by building random decision trees. Included to benchmark the autoencoder against a classical method and demonstrate the advantage of deep learning at scale.

---

## Key Features

**Log Parsing** — Converts raw unstructured log files into clean structured data using regex pattern matching. Extracts timestamps, log levels, component names and block IDs automatically.

**Event Sequencing** — Groups log entries by Block ID into fixed length sequences representing individual system operations. Captures the order of events not just their presence.

**Anomaly Detection** — Dual model approach using an Autoencoder neural network and Isolation Forest. Trained on the HDFS dataset containing 575,061 real log sequences with ground truth anomaly labels.

**Model Evaluation** — Detailed evaluation metrics including Precision, Recall and F1 score. Side by side comparison table between both models showing detected anomalies, correct detections and false alarms.

**Visualisation** — Three automatically generated charts saved after every run — log level distribution, anomaly scatter plot and sequence heatmap — giving administrators a clear visual picture of system health.

**Web Interface** — A Streamlit based web application allowing users to upload any log file and receive instant anomaly detection results with interactive visualisations. No command line required.

---

## Dataset
Trained and evaluated on the **HDFS (Hadoop Distributed File System)** log dataset from LogHub — a standard benchmark in log anomaly detection research containing over 11 million log lines from a real Hadoop cluster with 16,838 labelled anomalous block sessions.

---

## Tech Stack

| Purpose | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | TensorFlow / Keras |
| Machine Learning | scikit-learn |
| Data Processing | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Web Interface | Streamlit |
| Version Control | Git / GitHub |

---

## Results

| Model | Detected | Correct | False Alarms |
|---|---|---|---|
| Isolation Forest | 11,881 | 390 | 11,491 |
| Autoencoder | 556 | 471 | 85 |

The Autoencoder significantly outperforms Isolation Forest on the HDFS dataset, demonstrating that deep learning approaches are better suited for large scale log anomaly detection than traditional methods.

---

## Future Scope
- Real time log monitoring with live alerts
- Support for multiple log formats beyond HDFS
- Automated threshold tuning
- Email or Slack notifications when anomalies are detected
- LSTM based sequence modelling for improved temporal pattern detection
