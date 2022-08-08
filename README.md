# MachineLearning-model-with-cloud-database-demo
presentation link: https://drive.google.com/file/d/1Jo6KC8gtnQa6U2CwIr4Lox7xs1q8hIl_/view?usp=sharing


The final report is included in files.


I uploaded and managed the data in Amazon S3, then building the pipeline for reading image raw data from S3, extracting and constructing the images features (SIFT, color moments and GIST features etc.) using OpenCV and PySpark into Amazon RDS database, and providing api for downloading target image data and features data. Moreover, I trained the SVM classification model and stored in the S3 database for further frontend connection. The above work are implemented using Python.


The data ETL code can be found in fold 'spark_feature_extraction', the model relevant code can be found in fold "model_scripts".


This project was aim to develop a system that can predict a painting’s artist, style, and genre when the user input a picture of a painting. Specifically, the system uses several classification models trained on the existing artwork dataset and provides a user-friendly web page for users to browse the paintings in the existing artwork dataset and get information about a new painting. 


The data used in this project is artworks and corresponding information from WikiArt Dataset. We store data in Amazon S3 and extracted features from it using Spark. The created features and metadata are stored in database on Amazon RDS. The frontend webpage is realized based on Django and Bootstrap.

![image](https://user-images.githubusercontent.com/91628195/183343565-b33112ce-88c0-44e0-9447-a3b591e19880.png)



For prediction, we use SVM machine learning models and setting parameters by Grid Search.


This system’s functions include exploring artworks, uploading a new artwork to the database, extracting metadata and features, predicting an artwork’s style, genre or artist.


![image](https://user-images.githubusercontent.com/91628195/183363321-5d320563-cd76-42e0-b71f-54c6e5ef8c32.png)
\[Homepage\]

![image](https://user-images.githubusercontent.com/91628195/183363492-6fed45ff-1cbb-4340-8eb9-b6f27d2cbf89.png)
\[prediction result display\]

(More frontend display can be found in report.)

