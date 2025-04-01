# Spam Detection Model

## Overview
This project presents an advanced **Email/SMS Spam Classification System** leveraging **Natural Language Processing (NLP)** and **Machine Learning**. It efficiently processes input text, applies transformation techniques, and classifies messages as either **Spam** or **Not Spam** with high accuracy.

## Features
- Utilizes **TF-IDF vectorization** for precise text feature extraction.
- Implements **stemming** to enhance text normalization and reduce redundancy.
- Designed with an intuitive **Streamlit-based UI** for seamless user interaction.
- Employs a **Multinomial Naïve Bayes (MultinomialNB) model**, optimized for text classification.

## Project Structure
- **spam_detection (1).ipynb** - Jupyter Notebook documenting data preprocessing, model training, and evaluation methodologies.
- **app.py** - Streamlit application script for running the interactive spam classification tool.
- **model.pkl** - Serialized Multinomial Naïve Bayes model for prediction.
- **vectorizer.pkl** - TF-IDF vectorizer essential for text transformation.

## Installation & Dependencies
Ensure all required dependencies are installed before executing the application:
```bash
pip install streamlit nltk scikit-learn pandas numpy pickle5
```

## Usage Instructions
1. Clone the repository or download the necessary files.
2. Verify that **model.pkl** and **vectorizer.pkl** are placed in the same directory as **app.py**.
3. Execute the Streamlit application:
```bash
streamlit run app.py
```
4. Input an SMS or email message into the provided text field.
5. Click **Predict** to classify the message.

## Methodology
1. **Text Preprocessing**:
   - Converts text to lowercase to maintain uniformity.
   - Tokenizes words to facilitate further analysis.
   - Removes punctuation and stopwords to eliminate noise.
   - Applies **Porter Stemmer** for word normalization.
2. **Feature Extraction**:
   - Converts the preprocessed text into numerical features using **TF-IDF Vectorizer**.
3. **Prediction Mechanism**:
   - Utilizes a **Multinomial Naïve Bayes (MultinomialNB) classifier**, which is well-suited for text classification tasks.
   - The trained model analyzes the extracted features and determines if the message is **Spam** or **Not Spam**.

## Technologies Utilized
- **Python** (Core Programming Language)
- **Streamlit** (Interactive UI Framework)
- **NLTK** (Natural Language Processing Toolkit)
- **Scikit-learn** (Machine Learning Library)
- **Multinomial Naïve Bayes** (Text Classification Algorithm)

## Author
Developed by **Anand Kumar**

## License
This project is released under the **MIT License** and is open for further enhancements and contributions.
