# 📰 Arabic Fake News Detection

An Arabic Natural Language Processing (NLP) and Machine Learning project designed to classify Arabic news articles as **Real** or **Fake**.

The project includes Arabic text preprocessing, Exploratory Data Analysis (EDA), training and evaluation of multiple Machine Learning models, and an interactive Streamlit web application for analyzing Arabic news articles.

🚀 **Live Demo:** [Arabic Fake News Detection](https://arabic-fake-news-ai.streamlit.app/)

---

## ✨ Features

- Arabic text cleaning and preprocessing.
- Arabic diacritics removal and letter normalization.
- Arabic stop-words removal.
- Arabic text stemming.
- Exploratory Data Analysis (EDA).
- Data visualization using charts and tables.
- Training and comparison of three Machine Learning models.
- Automatic best-model selection based on Macro F1-score.
- TF-IDF feature extraction.
- Interactive Streamlit web application.
- Real/Fake news classification.
- Model confidence display when available.

---

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| NLP | NLTK, ISRIStemmer |
| Feature Extraction | TF-IDF |
| Machine Learning | Scikit-learn |
| Web Application | Streamlit |
| Model Serialization | Joblib |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
Fake_News_Detection-NLP/
│
├── app.py
├── run_pipeline.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── merged_cleaned.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       ├── .gitkeep
│       └── analysis/
│
├── models/
│   ├── best_model.joblib
│   ├── tfidf.joblib
│   └── results.csv
│
├── Notebook/
│   └── Fake_News_Detection__.ipynb
│
├── reports/
│   └── figures/
│       ├── article_length.png
│       ├── confusion_matrices.png
│       ├── label_distribution.png
│       └── model_comparison.png
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── analysis.py
    ├── train.py
    └── predict.py
```

---

## 📊 Dataset

The dataset contains **5,352 Arabic news articles**.

| Class | Number of Articles | Percentage |
|---|---:|---:|
| Real | 3,913 | 73.1% |
| Fake | 1,439 | 26.9% |
| **Total** | **5,352** | **100%** |

### Dataset Columns

- `Id`
- `date`
- `platform`
- `title`
- `News content`
- `Label`

### Class Imbalance

The dataset is imbalanced, with more Real articles than Fake articles.

Therefore, Macro F1-score is considered alongside accuracy because it treats both classes equally during evaluation.

---

## ⚙️ How It Works

The project follows a complete Machine Learning workflow:

### 1. Data Loading

Load the original dataset from:

```text
data/raw/merged_cleaned.csv
```

### 2. Exploratory Data Analysis (EDA)

Analyze the dataset to understand:

- Label distribution.
- Article length distribution.
- Publishing platforms.
- Common words and statistical insights.

### 3. Arabic Text Preprocessing

The preprocessing pipeline includes:

- Removing Arabic diacritics.
- Normalizing Arabic letters:
  - `أ`, `إ`, `آ` → `ا`
  - `ة` → `ه`
  - `ى` → `ي`
- Removing punctuation and numbers.
- Removing Arabic stop-words.
- Applying stemming using `ISRIStemmer`.

### 4. Feature Extraction

Convert preprocessed Arabic text into numerical representations using:

**TF-IDF (Term Frequency–Inverse Document Frequency)**

Maximum Features:

```python
MAX_FEATURES = 5000
```

### 5. Model Training

Train and compare three Machine Learning classifiers:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)

The dataset is divided using an 80/20 stratified train-test split.

### 6. Model Evaluation

Evaluate the trained models using performance metrics, including Macro F1-score.

### 7. Best Model Selection

Automatically select the best-performing model based on Macro F1-score.

The selected model and TF-IDF vectorizer are saved using Joblib.

### 8. News Prediction

Use the saved model to classify new Arabic news articles as Real or Fake through the Streamlit application.

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Directory

```bash
cd Fake_News_Detection-NLP
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Step 1: Run the Machine Learning Pipeline

```bash
python run_pipeline.py
```

The pipeline performs the following operations:

- Loads the raw dataset.
- Performs exploratory data analysis.
- Generates analysis tables and visualizations.
- Cleans and preprocesses Arabic text.
- Extracts TF-IDF features.
- Trains and evaluates the Machine Learning models.
- Selects and saves the best-performing model.

### Step 2: Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local application using the URL displayed in your terminal:

```text
http://localhost:8501
```

Alternatively, access the deployed application directly:

🌐 **[Try the Live Application](https://arabic-news-detector.streamlit.app/)**

---

## 🖥️ Streamlit Application

The project provides an interactive web application with four main sections:

| Section | Description |
|---|---|
| 🔍 Check News | Enter Arabic news text and receive a Real/Fake classification. |
| 📊 Data Analysis | Explore raw and processed datasets, charts, and statistical insights. |
| 🤖 Model Results | View model comparison results, performance charts, and confusion matrices. |
| ℹ️ About Project | Explore project objectives, technologies, and workflow. |

---

## 🧪 Model Results

The project compares three Machine Learning algorithms:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)

The evaluation results are saved in:

```text
models/results.csv
```

In our experiments, the models achieve approximately **88–90% accuracy**.

The Fake class is generally more challenging to detect because of the imbalance in the dataset.

Macro F1-score is also considered to provide a more balanced evaluation of both classes.

---

## 📈 Visualizations

The project generates several visualizations to better understand the dataset and evaluate model performance.

| Visualization | Description |
|---|---|
| Article Length Distribution | Shows the distribution of news article lengths. |
| Label Distribution | Shows the distribution of Real and Fake news articles. |
| Model Comparison | Compares the performance of trained models. |
| Confusion Matrices | Visualizes classification performance for each model. |

Generated figures are stored in:

```text
reports/figures/
```

---

## 🔧 Configuration

Project paths and configuration settings are managed in:

```text
src/config.py
```

Important settings include:

| Setting | Default | Description |
|---|---:|---|
| `TEST_SIZE` | 0.2 | Percentage of data used for testing. |
| `RANDOM_STATE` | 42 | Ensures reproducible results. |
| `MAX_FEATURES` | 5000 | Maximum number of TF-IDF features. |

---

## 💡 Future Improvements

Potential improvements include:

- Increasing the number of Fake News examples.
- Handling class imbalance using `class_weight='balanced'`.
- Experimenting with TF-IDF bigrams using `ngram_range=(1, 2)`.
- Exploring Arabic pre-trained language models such as AraBERT.
- Improving Fake News classification performance.
- Exploring deep learning approaches for Arabic text classification.

---

## ⚠️ Disclaimer

This project is developed for educational and learning purposes.

The model provides a Machine Learning prediction based on patterns learned from the training dataset. It does **not** perform independent fact-checking or guarantee the truthfulness of an article.

Always verify important news and information using reliable sources.

---

## 👨‍💻 Author

**Mohamed Ayman**

- GitHub: [MohamedAymanDev](https://github.com/MohamedAymanDev)
- LinkedIn: [Mohamed Ayman](https://www.linkedin.com/in/mohamed010ayman/)

---

⭐ If you find this project useful, consider giving it a star!