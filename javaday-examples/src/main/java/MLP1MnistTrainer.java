import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * A multi-layer perceptron with a single hidden layer.
 */

public class MLP1MnistTrainer {
    private static final Logger log = LoggerFactory.getLogger(MLP1MnistTrainer.class);
    static int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
    static int batchSize = 128; // How many examples to fetch with each step.
    static int numEpochs = 10; // An epoch is a complete pass through a given dataset.

    public static void main(String[] args) throws IOException {
        log.info("Load data....");
        DataSetIterator train = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator test = new MnistDataSetIterator(batchSize, true, 12345);

        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(28 * 28) // Number of input datapoints.
                .nOut(1000) // Number of output datapoints.
                .activation(Activation.RELU) // Activation function.
                .weightInit(WeightInit.XAVIER) // Weight initialization.
                .build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(1000)
                .nOut(10)
                .activation(Activation.SOFTMAX)
                .weightInit(WeightInit.XAVIER)
                .build())
            .pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork mlp1Net = new MultiLayerNetwork(conf);
        mlp1Net.init();

        Trainer.train(mlp1Net, train, test, "mlp1_mnist.zip");
        Trainer.printStats(test, mlp1Net);
    }
}
