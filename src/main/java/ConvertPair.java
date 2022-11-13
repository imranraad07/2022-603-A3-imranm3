import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;

public class ConvertPair extends ArrayWritable {
    public ConvertPair() {
        super(DoubleWritable.class);
    }

    public ConvertPair(Pair pair) {
        super(DoubleWritable.class, new DoubleWritable[]{pair.getIdx(), pair.getDist()});
    }
}
