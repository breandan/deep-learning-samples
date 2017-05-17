import java.util.Arrays;

public class Perceptron {
    public static void main(String[] args) {
        double[][] data = new double[][]{{0, 0, 0},{1, 1, 1}, {2, 1, 0}};
        double[] weights = new double[]{0, 0, 0};
        train(data, weights);

        System.out.println(classify(new double[]{0, 0}, weights));
        System.out.println(classify(new double[]{0, 1}, weights));
        System.out.println(classify(new double[]{2, 1}, weights));
        System.out.println(classify(new double[]{1, 1}, weights));

        System.out.println(Arrays.toString(weights));
    }

    private static void train(double[][] data, double[] weights) {
        //Training
        for(int e = 0; e < 100; e++) {
            double totalError = 0;

            for (double[] values : data) {
                double actual = values[2];
                double error = actual - classify(values, weights);
                totalError += Math.abs(error);
                weights[0] += error * 1;
                for (int i = 1; i < weights.length; i++) {
                    weights[i] += error * values[i-1];
                }
            }
            System.out.println("Total error: " + totalError);
            System.out.println(Arrays.toString(weights));
            if(totalError == 0)
                break;
        }
    }

    private static double classify(double[] value, double[] weights) {
        double pred = 1 * weights[0];
        for(int i = 1; i < weights.length; i++) {
            pred += weights[i] * value[i - 1];
        }

        if(pred < 0) {
            return 0;
        }

        return 1;
    }
}
