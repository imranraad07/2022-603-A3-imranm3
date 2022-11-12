import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class KNNMapper extends Mapper<Object, Text, IntWritable, CovertTwoDPair> {

    private int k;
    private Instances testInstances;
    private int testSize;
    private Pair[][] distMatrix;

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);

        Configuration conf = context.getConfiguration();

        k = conf.getInt("k", 0);
        testSize = conf.getInt("testNumInstances", 0);

        String path = new File("").getAbsolutePath();
        BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + conf.get("testSetPath")));
        ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);

        testInstances = arff.getData();
        reader.close();

        distMatrix = new Pair[testSize][k];

        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < k; j++) {
                distMatrix[i][j] = new Pair(-1, Double.MAX_VALUE);
            }
        }
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        if (value.toString().startsWith("@")) {
            return;
        }

        String[] valueTokens = value.toString().split(",");
        int trainAttrSize = valueTokens.length;

        double[] trainInstance = new double[trainAttrSize];

        for (int i = 0; i < trainAttrSize; i++) {
            trainInstance[i] = Double.parseDouble(valueTokens[i]);
        }

        for (int i = 0; i < testSize; i++) {
            double[] testInstance = testInstances.get(i).toDoubleArray();
            double dist = calculateDistance(trainInstance, testInstance);

            Pair distPair = new Pair();
            boolean minDistFlag = false;

            for (int j = 0; j < k; j++) {
                Pair tmpPair = new Pair(distMatrix[i][j]);

                if (!minDistFlag) {
                    if (dist < tmpPair.getDist().get()) {
                        minDistFlag = true;
                        distMatrix[i][j].copyValuesFrom(trainInstance[trainAttrSize - 1], dist);
                        distPair = new Pair(tmpPair);
                    }
                } else {
                    distMatrix[i][j].copyValuesFrom(distPair);
                    distPair.copyValuesFrom(tmpPair);
                }
            }
        }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        IntWritable outputKey = new IntWritable(context.getTaskAttemptID().getTaskID().getId());
        ConvertPair[][] output = new ConvertPair[testSize][k];

        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < k; j++) {
                output[i][j] = new ConvertPair(distMatrix[i][j]);
            }
        }

        CovertTwoDPair outputMatrix = new CovertTwoDPair(output);
        context.write(outputKey, outputMatrix);
        super.cleanup(context);
    }

    private double calculateDistance(double[] a, double[] b) {
        double sum = 0;
        double diff;

        for (int i = 0; i < a.length - 1; i++) {
            diff = a[i] - b[i];
            sum += diff * diff;
        }

        return sum;
    }
}
