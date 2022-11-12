import org.apache.hadoop.io.TwoDArrayWritable;

public class CovertTwoDPair extends TwoDArrayWritable {
    public CovertTwoDPair() {
        super(ConvertPair.class);
    }

    public CovertTwoDPair(ConvertPair[][] values) {
        super(ConvertPair.class, values);
    }

}
