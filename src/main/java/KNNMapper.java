import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class KNNMapper extends Mapper<Object, Text, IntWritable, CovertTwoDPair> {

    private int k;
    private Instances testInstances;
    private int testNumInstances;
    private Pair[][] distMatrix;

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);

        Configuration configuration = context.getConfiguration();

        k = configuration.getInt("k", 0);
        testNumInstances = configuration.getInt("testNumInstances", 0);

        String path = new File("").getAbsolutePath();
        BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + configuration.get("testSetPath")));
        ArffReader arff = new ArffReader(reader);
        testInstances = arff.getData();
        reader.close();

        distMatrix = new Pair[testNumInstances][k];

        for (int i = 0; i < testNumInstances; i++) {
            Arrays.fill(distMatrix[i], new Pair(-1, Double.MAX_VALUE));
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
        AtomicInteger index = new AtomicInteger();
        Arrays.stream(value.toString().split(",")).forEach(s -> trainInstance[index.getAndIncrement()] = Double.parseDouble(s));

        int i = 0;
        for (Instance instance : testInstances) {
            double dist = calculateDistance(trainInstance, instance.toDoubleArray());

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
            i++;
        }
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        IntWritable outputKey = new IntWritable(context.getTaskAttemptID().getTaskID().getId());
        ConvertPair[][] output = new ConvertPair[testNumInstances][k];

        for (int i = 0; i < testNumInstances; i++) {
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
