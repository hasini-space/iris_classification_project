# Iris Flower Classification Project

## 🌸 Overview

This project implements a machine learning classification system for the famous **Iris dataset**. The goal is to build and train models that can accurately classify iris flowers into three species:[...]

## 📊 Dataset

The Iris dataset contains 150 flower samples with 4 features each:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

Each sample is labeled with one of three iris species.

## 🎯 Project Goals

- ✅ Explore and analyze the Iris dataset
- ✅ Preprocess and prepare data for modeling
- ✅ Build multiple classification models
- ✅ Evaluate model performance using various metrics
- ✅ Compare different algorithms and select the best performer
- ✅ Provide predictions on new data

## 📚 Technologies & Libraries

- **Python 3.x**
- **scikit-learn** - Machine Learning algorithms
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib / Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development
- **Streamlit** - Web application framework

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/hasini-space/iris_classification_project.git
cd iris_classification_project

# Install required dependencies
pip install -r requirements.txt
```

## 📖 Usage

1. **Load and Explore Data**
   ```python
   python exploratory_analysis.py
   ```

2. **Train Models**
   ```python
   python train_models.py
   ```

3. **Make Predictions**
   ```python
   python predict.py --features "5.1, 3.5, 1.4, 0.2"
   ```

## 📁 Project Structure

```
iris_classification_project/
├── data/
│   └── iris.csv                 # Dataset
├── notebooks/
│   └── analysis.ipynb           # Exploratory analysis
├── src/
│   ├── train_models.py          # Model training
│   ├── predict.py               # Prediction script
│   └── utils.py                 # Utility functions
├── models/
│   └── best_model.pkl           # Trained model
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🤖 Models Implemented

- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**

## 📈 Results

Model performance comparison (Accuracy, Precision, Recall, F1-Score):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | - | - | - | - |
| Decision Tree | - | - | - | - |
| Random Forest | - | - | - | - |
| SVM | - | - | - | - |
| KNN | - | - | - | - |

*Results to be updated after model training*

## 📊 Visualizations

- Feature distributions and correlations
- Confusion matrices for each model
- ROC curves and AUC scores
- Feature importance plots

## 🚀 Live Deployment

### Streamlit Web Application
- **Live Demo**: [Iris Flower Classification Dashboard](https://irisclassificationproject-gt6g5m57zjvmxn8uwhappgx.streamlit.app/)

To deploy locally with Streamlit:
```bash
streamlit run app.py
```

## 📝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m "Add improvement"`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Hasini Space**  
GitHub: [@hasini-space](https://github.com/hasini-space)

## 📞 Contact & Support

For questions or issues, please:
- Open an [Issue](https://github.com/hasini-space/iris_classification_project/issues)
- Submit a Pull Request with improvements
- Reach out via GitHub Discussions

## 📚 References

- [Iris Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [scikit-learn Official Documentation](https://scikit-learn.org/)
- [Machine Learning Basics](https://scikit-learn.org/stable/modules/tree.html)

---

**Last Updated**: June 2026  
**Status**: 🔄 In Development
