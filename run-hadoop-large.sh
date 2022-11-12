rm -R predictions
hadoop jar target/KNN-MapReduce-0.0.1-SNAPSHOT-jar-with-dependencies.jar KNN datasets/large-train.arff datasets/large-test.arff 3 predictions