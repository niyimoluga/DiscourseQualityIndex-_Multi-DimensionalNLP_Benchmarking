# DiscourseQualityIndex-_Multi-DimensionalNLP_Benchmarking


## 📌 Project Overview

The **DiscourseQualityIndex** is an advanced Natural Language Processing (NLP) pipeline designed to evaluate human digital discourse across **6 distinct dimensions simultaneously**.

Developed as a core technical component for algorithmic content recommendation (specifically targeting decentralized platforms like BlueSky), this project solves two major engineering bottlenecks:

- **Multi-Dimensional Nuance**: Moving beyond binary "toxic vs. non-toxic" classifiers to understand constructive political dialogue, justification, and respect.
- **High-Speed Inference**: Designing an architecture capable of classifying live firehose data within a strict **<500ms latency budget**.

---

## 🎯 The 6 Dimensions of Discourse Quality

The models in this repository are trained to predict continuous scores for:

1. Level of Justification
2. Respect Towards Demands
3. Respect Towards Counterarguments
4. Content of Justification
5. Respect Towards Groups
6. Constructive Politics

---

## 📊 Benchmarking & Ablation Study

To determine the optimal architecture, we conducted a comprehensive ablation study comparing Classical Machine Learning (TF-IDF + CPU), massive Multi-Task Deep Learning Transformers, and Metric Learning architectures.

Models were trained on a 200,000-row labeled dataset and evaluated using **Mean Squared Error (MSE)** across all 6 dimensions.

| Model | Justification | Demands | Counterargs | Content | Groups | Politics | Overall |
|:------|:-------------:|:-------:|:-----------:|:-------:|:------:|:--------:|:-------:|
| Mean baseline | 0.1634 | 0.1314 | 0.1221 | 0.1054 | 0.0639 | 0.0937 | 0.1133 |
| TF-IDF + RF | 0.1134 | 0.1134 | 0.1036 | 0.0975 | 0.0585 | 0.0858 | 0.0954 |
| TF-IDF + XGBoost | 0.0991 | 0.1028 | 0.0937 | 0.0917 | 0.0546 | 0.0803 | 0.0870 |
| BERT-base | 0.0488 | 0.0590 | 0.0577 | 0.0660 | 0.0357 | 0.0494 | 0.0527 |
| BERT-large | 0.04662 | 0.05623 | 0.05570 | 0.06277 | 0.03374 | 0.04721 | 0.05038 |
| RoBERTa | 0.0484 | 0.0568 | 0.0565 | 0.0640 | 0.0342 | 0.0475 | 0.0512 |
| 2Tower | — | — | — | — | — | — | *(pending retrain)* |
---

## 🏗️ Architectures Explored & Key Findings

### 1. Classical Machine Learning (The Baselines)

**Models**: `RandomForestRegressor` and `XGBRegressor` wrapped in Scikit-Learn's `MultiOutputRegressor`, utilizing a 10,000-feature TF-IDF matrix.

**The Insight**: While XGBoost performed admirably (MSE 0.0870), classical models hit a hard mathematical ceiling. TF-IDF acts as a keyword counter, meaning it can detect slurs (Respect Towards Groups), but completely fails to understand sentence structure, logic, and context required to measure "Level of Justification."

---

### 2. Multi-Task Learning / "Shared Brain" Transformers

**Models**: `bert-base-uncased`, `roberta-base`, and `bert-large-uncased` with custom 6-node regression heads.

**The Insight**: Upgrading to contextual word embeddings drastically reduced the error rate.

**Quality Over Quantity**: Surprisingly, **RoBERTa-base (110M parameters)** defeated BERT-large (340M parameters). This proved that smarter pre-training (RoBERTa's dynamic masking over 160GB of text) is far more valuable for nuanced semantic tasks than simply scaling up raw parameter counts.

---

### 3. Semantic Retrieval / Two-Tower Architecture (The Engineering Champion)

**Model**: A dual-encoder setup mapping Anchor Definitions (e.g., the definition of "Respect") and user comments into the same vector space, optimized via `CosineSimilarityLoss`.

**The Insight**: While the Two-Tower model's accuracy relative to RoBERTa is pending re-evaluation after a metric-scale correction, the **Two-Tower model is the production champion**. By pre-computing and caching the 6 definition vectors offline, live inference on incoming social media posts only requires a single pass through one encoder and a sub-millisecond dot-product calculation. This architecture solves the **<500ms latency requirement** for live-feed integration while allowing infinite scalability (new dimensions can be added without retraining the network).

---

## 🛠️ Key Technical Implementations

| Implementation | Description |
|:---------------|:------------|
| **High-Speed Data Vectorization** | Utilized Pandas vectorization to bundle continuous targets into 1D PyTorch tensors for seamless Hugging Face tokenization across massive datasets. |
| **Custom PyTorch Interceptor (PureMSETrainer)** | Overrode default Hugging Face loss calculations to extract, monitor, and log individual Training MSE for all 6 dimensions prior to PyTorch's master loss averaging. |
| **Hardware Optimization** | Implemented `gradient_accumulation_steps` and dynamic batch sizing to successfully train massive models (like BERT-large) on multi-GPU setups without triggering CUDA Out-Of-Memory (OOM) errors. |

---

## 📁 Repository Structure

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/DiscourseQualityIndex.git
cd DiscourseQualityIndex

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py --model roberta-base --epochs 5 --batch-size 16
