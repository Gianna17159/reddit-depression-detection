# Detecting Symptoms of Depression on Reddit

This project uses Reddit posts from depression and mental health related subreddits to detect symptoms of depression.
Steps include:

1. Identified ~94k symptom posts from subreddits associated with depression symptoms, then formed a control dataset using posts by those same users in non-depression subreddits.
2. Processed all posts by removing most common 100 words in the control dataset.
3. Trained an LDA model on control and symptom datasets to generate 200 topics for each post, and stored the topic probabilities for each post.
4. Used a pretrained distilRoberta model to generate embeddings, using the average of the fifth layer output to represent each post.
5. Evaluated the performance of distilRoberta and LDA models using 5-fold cross validation with random forest classifier. The random forest classifier was one-to-one, meaning there were 10 models, one for each symptom dataset, which were evaluated by combining each symptom dataset with the control dataset.

To run this code for the first time, run all cells in order. This project makes use of pickle.load() and pickle.dump() to store original dataset (student.pkl), unprocessed symptom and control datasets (control_dict.pkl, depression_dict.pkl), processed datasets (control_preprocessed.pkl, depression_preprocessed.pkl), combined control+symptom dataset with topic probabilities (after_lda_df.pkl), and combined control+symptom dataset with topic probabilities and embeddings (after_roberta_df.pkl). If all pickle files are downloaded, cells marked with "ONLY RUN ONCE" do not need to be run. AUC scores for LDA and Roberta evaluation are produced with calls to main in the last two cells, and are printed as output.
