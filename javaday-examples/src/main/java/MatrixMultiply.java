import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Scanner;

public class MatrixMultiply {
    public static void main(String args[]) {
        multiplyMatrices();
    }

    private static double[][] loop(double[][] first, double[][] second) {
        double multiply[][] = new double[first.length][second[0].length];
        for (int c = 0; c < first.length; c++) {
            for (int d = 0; d < second[0].length; d++) {
                double sum = 0;

                for (int k = 0; k < second.length; k++) {
                    sum = sum + first[c][k] * second[k][d];
                }

                multiply[c][d] = sum;
            }
        }

        return multiply;
    }

    private static INDArray nd4j(double[][] first, double[][] second) {
        return Nd4j.create(first).mmul(Nd4j.create(second));
    }

    private static void multiplyMatrices() {
        int m, n, p, q, c, d;

        Scanner in = new Scanner(System.in);
        System.out.println("Enter the number of rows and columns of matrix #1");
        m = in.nextInt();
        n = in.nextInt();

        double first[][] = new double[m][n];

        System.out.println("Enter data for matrix #1");

        for (c = 0; c < m; c++)
            for (d = 0; d < n; d++)
                first[c][d] = in.nextInt();

        System.out.println("Enter data for matrix #2");
        p = in.nextInt();
        q = in.nextInt();

        if (n != p)
            System.out.println("Input dimensions incompatible.");
        else {
            double second[][] = new double[p][q];

            System.out.println("Enter the elements of second matrix");

            for (c = 0; c < p; c++)
                for (d = 0; d < q; d++)
                    second[c][d] = in.nextInt();

            double[][] multiply = loop(first, second);

            System.out.println("Imperative result:");

            for (c = 0; c < m; c++) {
                for (d = 0; d < q; d++)
                    System.out.print(multiply[c][d] + "\t");

                System.out.print("\n");
            }

            System.out.println("ND4J inner product result:");
            System.out.println(nd4j(first, second));
        }
    }
}
