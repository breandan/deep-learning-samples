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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

public class MLP2MnistTrainer {
    private static final Logger log = LoggerFactory.getLogger(MLP2MnistTrainer.class);
    static int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
    static int batchSize = 128; // How many examples to fetch with each step.
    static int numEpochs = 10; // An epoch is a complete pass through a given dataset.

    public static void main(String[] args) throws IOException {
        final int numRows = 28; // The number of rows of a matrix.
        final int numColumns = 28; // The number of columns of a matrix.
        int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
        int batchSize = 128; // How many examples to fetch with each step.
        int rngSeed = 123; // This random-number generator applies a seed to ensure that the same initial weights are used when training. We’ll explain why this matters later.
        int numEpochs = 15; // An epoch is a complete pass through a given dataset.

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, true, 12345);

        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .activation(Activation.LEAKYRELU)
            .weightInit(WeightInit.XAVIER)
            .learningRate(0.02)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
            .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
            .pretrain(false).backprop(true)
            .build();

        MultiLayerNetwork mlpNet = new MultiLayerNetwork(conf);
        mlpNet.init();

        train(mlpNet, mnistTrain, mnistTest);
        printStats(mnistTest, mlpNet);
    }

    private static void printStats(DataSetIterator mnistTest, MultiLayerNetwork net) {
        //Perform evaluation (distributed)
        Evaluation evaluation = net.evaluate(mnistTest);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("***** Example Complete *****");
    }

    private static void train(MultiLayerNetwork model,
                              DataSetIterator train,
                              DataSetIterator test) throws IOException {
        //Save the model
        File locationToSave = new File("jfokus-examples/src/main/resources/mlp1_mnist.zip");
        //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;
        //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc.
        //Save this if you want to train your network more in the future

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(1));
        for (int i = 0; i < numEpochs; i++) {
            model.fit(train);
            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while (test.hasNext()) {
                DataSet ds = test.next();
                INDArray output = model.output(ds.getFeatureMatrix(), false);
                eval.eval(ds.getLabels(), output);
            }

            ModelSerializer.writeModel(model, locationToSave, saveUpdater);

            log.info(eval.stats());
            test.reset();
        }
    }
}
