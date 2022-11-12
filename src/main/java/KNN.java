import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Scanner;

public class KNN {

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Error in Arguments: " +
                    "hadoop jar target/KNN-MapReduce-0.0.1-SNAPSHOT-jar-with-dependencies.jar " +
                    "KNN datasets/large-train.arff datasets/large-test.arff 3 predictions");
            System.exit(2);
        }

        Configuration conf = new Configuration();

        conf.set("testSetPath", args[1]);
        conf.setInt("k", Integer.parseInt(args[2]));

        String path = new File("").getAbsolutePath();
        BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[1]));
        ArffReader arff = new ArffReader(reader);
        Instances data = arff.getData();

        int numInstances = data.numInstances();
        conf.setInt("testNumInstances", numInstances);

        Job job = Job.getInstance(conf, "KNN");

        job.setJarByClass(KNN.class);
        job.setMapperClass(KNNMapper.class);
        job.setReducerClass(KNNReducer.class);

        job.setNumReduceTasks(1);

        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(CovertTwoDPair.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[3]));

        long startTime = System.currentTimeMillis();

        job.waitForCompletion(true);

        long endTime = System.currentTimeMillis();

        long milliseconds = (endTime - startTime);

        File file = new File(path + System.getProperty("file.separator") + "predictions/part-r-00000");
        Scanner sc = new Scanner(file);

        reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[1]));
        arff = new ArffReader(reader, 1000);
        data = arff.getData();
        data.setClassIndex(data.numAttributes() - 1);

        Instance instance;
        int correct = 0;

        while (sc.hasNextLine() && (instance = arff.readInstance(data)) != null) {
            sc.nextInt();
            int prediction = sc.nextInt();

            if (prediction == (int) instance.classValue()) {
                correct++;
            }
        }
        sc.close();
        reader.close();

        double accuracy = (double) correct / numInstances;

        System.out.println("It took " + milliseconds + " ms and results have an accuracy of " + accuracy);
    }
}
