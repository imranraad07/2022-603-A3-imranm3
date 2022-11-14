import org.apache.hadoop.io.DoubleWritable;

public class Pair {
    DoubleWritable idx;
    DoubleWritable dist;

    Pair(DoubleWritable idx, DoubleWritable dist) {
        this.dist = dist;
        this.idx = idx;
    }

    public Pair() {
    }

    public Pair(Pair anotherPair) {
        this.dist = anotherPair.getDist();
        this.idx = anotherPair.getIdx();
    }


    public Pair(double key, double dist) {
        this.dist = new DoubleWritable(dist);
        this.idx = new DoubleWritable(key);
    }


    public DoubleWritable getIdx() {
        return idx;
    }

    public void setIdx(DoubleWritable idx) {
        this.idx = idx;
    }

    public DoubleWritable getDist() {
        return dist;
    }

    public void setDist(DoubleWritable dist) {
        this.dist = dist;
    }

    // copy another pair to current pair
    public void copyValuesFrom(Pair anotherPair) {
        this.setIdx(anotherPair.getIdx());
        this.setDist(anotherPair.getDist());
    }

    public void copyValuesFrom(double key, double dist) {
        this.dist = new DoubleWritable(dist);
        this.idx = new DoubleWritable(key);
    }

}
