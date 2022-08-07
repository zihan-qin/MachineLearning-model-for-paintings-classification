Spark version: 3.1.2
scala version: 2.12.10
hadoop: 3.2


- extract_feature.py

spark script for updating features given sample name by the system argument

input: name

output: None (update features in RDS features table)

- spark_connection_utils.py

utils for spark to connect to AWS S3 and RDS

SparkRDS:

upload_features: input a Features namedtuple and upload it to RDS features table

download_features: given a sample name, download the feature in return_type(Spark Rowor dict)

get_path: given a sample name, return url path of it

SparkS3:

download_image: input a url path, return the image in spark image formation

- image_process.py

functions for extracting image features: sift, color_moments, color_gist

- load_features.py (deprecated)

dump all S3 images that can be found in the RDS train_data table into features and store them in RDS features table
