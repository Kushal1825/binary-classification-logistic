# Breast Cancer Classification using Logistic Regression

## 📌 Objective
The goal of this project is to build a **binary classifier** using **Logistic Regression** to predict whether a tumor is **Malignant (M)** or **Benign (B)**, based on the **Breast Cancer Wisconsin Dataset**.

This task demonstrates:
- Binary classification with logistic regression  
- Feature preprocessing and standardization  
- Model evaluation using **confusion matrix, precision, recall, ROC-AUC**  
- Understanding the **sigmoid function** and threshold tuning  

---

## 📊 Dataset
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source**: Provided as `data.csv`  
- **Shape**: 569 rows × 31 features (+ 1 target column `diagnosis`)  
- **Target Variable**:  
  - `M` → Malignant (encoded as `1`)  
  - `B` → Benign (encoded as `0`)  

Columns include various computed features of cell nuclei (e.g., `radius_mean`, `texture_mean`, etc.).

---

## ⚙️ Steps Performed
1. **Data Loading & Cleaning**
   - Removed unnecessary columns (`id`, `Unnamed: 32`)  
   - Encoded target labels (`M=1`, `B=0`)  

2. **Preprocessing**
   - Standardized features using **StandardScaler**  

3. **Train/Test Split**
   - 70% training, 30% testing  

4. **Model Training**
   - Trained a **Logistic Regression** model (`max_iter=500`)  

5. **Evaluation Metrics**
   - Confusion Matrix  
   - Precision, Recall, F1-score  
   - ROC-AUC score and ROC Curve  

6. **Threshold Tuning**
   - Adjusted sigmoid cutoff values (0.4, 0.5, 0.6) to study effect on predictions  

---

## 📈 Results
- The model achieved **high accuracy and ROC-AUC score**.  
- Confusion matrix and classification report indicate that logistic regression is effective for this dataset.  
- ROC Curve shows strong separation between classes.  

---

## ▶️ How to Run

### 1. Clone this repository  
```bash
git clone https://github.com/Kushal1825/binary-classification-logistic.git
cd binary-classification-logistic
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the script
```bash
python index.py
```

