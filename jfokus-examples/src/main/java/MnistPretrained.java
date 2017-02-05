import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.util.*;

public class MnistPretrained {
    public static void main(String[] args) throws IOException, InterruptedException {
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();
        MultiLayerNetwork lenet = ModelSerializer.restoreMultiLayerNetwork(new File(classLoader.getResource("lecun_mnist").getFile()));
        MultiLayerNetwork mlp1 = ModelSerializer.restoreMultiLayerNetwork(new File(classLoader.getResource("mlp1_mnist").getFile()));
        MultiLayerNetwork mlp2 = ModelSerializer.restoreMultiLayerNetwork(new File(classLoader.getResource("mlp2_mnist").getFile()));
        MultiLayerNetwork sae = ModelSerializer.restoreMultiLayerNetwork(new File(classLoader.getResource("sae_mnist").getFile()));

        List<String> labels = new ArrayList<>(10);
        for(int i = 0; i < 10; i++) {
            labels.add(i + "");
        }

        RecordReader recordReader = new ImageRecordReader(28, 28, 1);
        recordReader.initialize(new FileSplit(new File(classLoader.getResource("./").getFile())));

        DataSetIterator dataSet = new RecordReaderDataSetIterator(recordReader, 1);

        while(dataSet.hasNext()) {
            INDArray indArray = dataSet.next().getFeatureMatrix();
            float[] f = indArray.data().asFloat();
            float[] t = new float[f.length];
            for(int i = 0; i < f.length; i++) {
                if(f[i] < 255)
                    t[i] = 1;
                else
                    t[i] = 0;
            }
            indArray = Nd4j.create(t);

            INDArray lenetOutput = lenet.output(indArray);
            INDArray mlp1Output = mlp1.output(indArray);
            INDArray mlp2Output = mlp2.output(indArray);

            System.out.println(lenetOutput + " <- LeNet (Convolutional Neural Net)");
            System.out.println(mlp1Output + " <- 1-Layer Multilayer Perceptron");
            System.out.println(mlp2Output + " <- 2-Layer Multilayer Perceptron");

            System.out.println("   ^     ^     ^     ^     ^     ^     ^     ^     ^     ^");
            System.out.println("   0     1     2     3     4     5     6     7     8     9");
        }
    }
}
