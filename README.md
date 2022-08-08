# MachineLearning-model-with-cloud-database-demo
presentation link: https://drive.google.com/file/d/1Jo6KC8gtnQa6U2CwIr4Lox7xs1q8hIl_/view?usp=sharing
The final report is included in files.

This project was aim to develop a system that can predict a painting’s artist, style, and genre when the user input a picture of a painting. Specifically, the system uses several classification models trained on the existing artwork dataset and provides a user-friendly web page for users to browse the paintings in the existing artwork dataset and get information about a new painting. 


The data used in this project is artworks and corresponding information from WikiArt Dataset. We store data in Amazon S3 and extracted features from it using Spark. The created features and metadata are stored in database on Amazon RDS. The frontend webpage is realized based on Django and Bootstrap.

![image](https://user-images.githubusercontent.com/91628195/183343565-b33112ce-88c0-44e0-9447-a3b591e19880.png)

I collected the data from WikiArt extract three types of features from the images, which are SIFT, color moments and GIST features. Besides, OpenCV is employed for extracting metadata of images, such as height, width, depth and mode.


For prediction, we use SVM machine learning model and setting parameters by Grid Search.



