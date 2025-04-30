# Trip-Advisor-NLP-Analysis
The goal of this project is to perform sentiment analysis on Tripadvisor hotel reviews by developing & comparing different ML models to classify reviews based on their textual content and corresponding star ratings (out of 5).

Google Colab was used to host our code, along with the aid of various NLP and ML libraries such as scikit-learn, nltk, and sentence transformers.

In order to run the code, first download the dataset found on Kaggle https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews/data and name it 'tripadvisor_hotel_reviews.csv.' 
Download all three of the ipynb files found in the repository and open them up in three separate Google Colab notebooks. 
Connect to the T4 GPU by selecting "Change runtime type," selecting the T4 GPU, and pressing "Save."
Upload the dataset to each of these notebooks to ensure the code works appropriately. 
After uploading the dataset to the notebook, click on "Runtime" at the top of the page and hit "Run all." You can also run each cell indiviually.

The file NLPFinalProjectEDAVisualizations.ipynb contains the exploratory data analysis, label mapping, pre-processing, and splitting/tokenization sections of our project. 

The file NLP_Final_Count&TFIDF.ipynb performs vectorization using CountVectorizer and TFIDF, both with max_features=5000. Then, Random Forest and K-Nearest Neighbor classifiers to train our models. Finally, the models are tested and evaluated using classification reports, a confusion matrix, an ROC + AUC curve graph, and LIME examples.

The file NLP_Final_Project_SentenceTransformers.ipynb performs SentenceTransformer vectorization using the distilbert-base-uncased model. Then, Random Forest and K-Nearest Neighbor classifiers to train our models. Finally, the models are tested and evaluated using classification reports.
