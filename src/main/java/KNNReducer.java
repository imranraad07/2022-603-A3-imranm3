import com.sun.org.apache.bcel.internal.generic.SWAP;
import com.sun.org.apache.xml.internal.security.utils.JavaUtils;
import jdk.jfr.internal.Utils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import weka.filters.unsupervised.attribute.SwapValues;

import java.io.IOException;
import java.sql.Wrapper;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

public class KNNReducer extends Reducer<IntWritable, CovertTwoDPair, IntWritable, IntWritable> {
    private int k;
    private int testSize;

    private Pair[][] distMatrix;

    @Override
    public void setup(Context context) throws IOException, InterruptedException {
        super.setup(context);

        Configuration conf = context.getConfiguration();

        k = conf.getInt("k", 0);
        testSize = conf.getInt("testNumInstances", 0);

        distMatrix = new Pair[testSize][k];

        for (int i = 0; i < testSize; i++) {
            for (int j = 0; j < k; j++) {
                distMatrix[i][j] = new Pair(-1, Double.MAX_VALUE);
            }
        }
    }

    @Override
    public void reduce(IntWritable key, Iterable<CovertTwoDPair> value, Context context) throws IOException, InterruptedException {
        value.forEach(covertTwoDPair -> {
            ConvertPair[][] matrix = (ConvertPair[][]) covertTwoDPair.toArray();

            for (int i = 0; i < testSize; i++) {
                for (int j = 0; j < k; j++) {
                    Pair current = new Pair((DoubleWritable) matrix[i][j].get()[0], (DoubleWritable) matrix[i][j].get()[1]);
                    Pair tmp = new Pair();

                    if (current.getDist().get() < distMatrix[i][j].getDist().get()) {
                        tmp.copyValuesFrom(distMatrix[i][j]);
                        distMatrix[i][j].copyValuesFrom(current);

                        for (int m = j + 1; m < k; m++) {
                            Pair tmp2 = new Pair(distMatrix[i][m]);
                            distMatrix[i][m].copyValuesFrom(tmp);
                            tmp.copyValuesFrom(tmp2);
                        }
                    }
                }
            }
        });
    }

    @Override
    public void cleanup(Context context) throws IOException, InterruptedException {
        for (int i = 0; i < testSize; i++) {
            Map<Integer, Integer> classCounts = new HashMap<Integer, Integer>();

            for (int j = 0; j < k; j++) {
                Pair classDistPair = new Pair(distMatrix[i][j]);

                int currentClass = (int) classDistPair.getIdx().get();

                if (!classCounts.containsKey(currentClass)) {
                    classCounts.put(currentClass, 0);
                } else {
                    classCounts.put(currentClass, classCounts.get(currentClass) + 1);
                }
            }

            final int[] predictedClass = {-1};
            final int[] highestClassCount = {0};

            classCounts.entrySet().stream().forEach(entry -> {
                int key = entry.getKey();
                int val = entry.getValue();

                if (val > highestClassCount[0]) {
                    predictedClass[0] = key;
                    highestClassCount[0] = val;
                }
            });

            context.write(new IntWritable(i), new IntWritable(predictedClass[0]));
        }

        super.cleanup(context);
    }
}
