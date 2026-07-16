# 🧬: IDF-EC: Interpretable Dynamic Feature–Logit Fusion for Enzyme Commission Number Prediction

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Paper](https://img.shields.io/badge/17th%20ACM%20BCB%20-Accepted-orange.svg)

---

## 📖 Abstract

Accurate Enzyme Commission (EC) number annotation is fundamental to functional genomics, metabolic modeling, and enzyme discovery.
A key drawback of existing EC number prediction approaches is that each model architecture performs inconsistently across different EC classes and hierarchical levels, making it challenging for biologists to reach reliable conclusions.
Ensemble learning offers a promising solution by leveraging the complementary strengths of multiple base learners.
Nevertheless, aggregating several predictors typically reduces model interpretability, which is crucial for biological validation and for gaining functional insight.
In this study, we present a novel deep learning ensemble framework, named IDF-EC, that integrates heterogeneous state-of-the-art EC number deep learning models while preserving residue-level interpretability.
IDF-EC dynamically combines ECPICK, CLEAN, and HIT-EC through an adaptive fusion strategy designed to leverage complementary predictive signals while retaining model-level explanations.
We evaluated IDF-EC on 237,477 curated protein sequences spanning 2,445 EC numbers using repeated stratified cross-validation.
Across all four EC levels, IDF-EC consistently outperformed individual models and conventional ensemble strategies.
It achieved statistically significant improvements in both micro- and macro-averaged F1-scores, demonstrating robust gains across diverse enzyme classes.
Importantly, the framework preserved the interpretability mechanisms of the base learners, enabling the localization of functionally relevant sequence regions and providing biologically meaningful insights alongside improved predictive accuracy.

---

## 🧰 Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/datax-lab/IDF-EC.git
cd IDF-EC
pip install -r requirements.txt
```
---

## 🌐 Online Model Predictions & Interpretations

You can use IDF-EC directly through our interactive web platform:

👉 **Website:** [https://enzymex.dataxlab.org/](https://enzymex.dataxlab.org/)

The website allows you to:

* Upload FASTA sequences
* Run IDF-EC predictions
* Visualize contribution scores

---

## 🧪 Training Your Own Model

1. Train ECPICK : [ECPICK](https://github.com/datax-lab/ECPICK)

2. Train CLEAN : [CLEAN](https://github.com/tttianhao/CLEAN)

3. Train HIT-EC : [HIT-EC](https://github.com/datax-lab/HIT-EC)

4. Train IDF-EC
```
python train_idfec.py --gate False
```
Choose the best model to train the dynamic fusion gate.
```
python train_idfec.py --gate True --model your_model
```


## 🚀 Quick Demo

You can use the **Demo notebook** to predict and interpret the model's predictions.

---

## 📁 Repository Structure

```
IDF-EC/
│
├── models/             # ECPICK, CLEAN, HIT-EC and IDF-EC
│   ├── ECPICK.py
│   ├── CLEAN.py
│   ├── HIT-EC.py
│   └── IDF-EC.py
├── utils/              # Label encoder and util functions
│   ├── utils.py
│   ├── output_classes.pkl          
│   └── tokenizer.pickle             
├── esm_data/           # Directory for embedding features from ESM-1b
├── data/               # Dataset directory for models
│   ├── ECPICK
│   ├── CLEAN
│   ├── HIT-EC
│   └── dataset.csv
├── saved_models        # Directory for IDF-EC pth files
│   └── final_epoch=2.pth
├── clean_inference     # Directory for CLEAN inference results
├── base_learners       # Directory for base learners weight files
│   ├── ECPICK_models
│   ├── CLEAN_models
│   └── HIT-EC_models
├── demo_interpretation # Directory for interpretation files and images
├── train_idfec.py      # Training python file for IDF-EC
├── dataset.ipynb       # Data split for training
├── Demo.ipynb          # Inference and Interpretation Demo of IDF-EC
└── extract.py          # ESM-1b script
```

---

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

---

**Contact:** suhyeong.jeon@unlv.edu  
**Maintained by:** DataX-Lab, UNLV
