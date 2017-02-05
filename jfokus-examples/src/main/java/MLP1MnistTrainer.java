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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

import static org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class MLP1MnistTrainer {
    private static final Logger log = LoggerFactory.getLogger(MLP1MnistTrainer.class);
    static int outputNum = 10; // Number of possible outcomes (e.g. labels 0 through 9).
    static int batchSize = 128; // How many examples to fetch with each step.
    static int numEpochs = 10; // An epoch is a complete pass through a given dataset.

    public static void main(String[] args) throws IOException {
        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, true, 12345);

        //----------------------------------
        //Create network configuration and conduct network training
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
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

    public static void train(MultiLayerNetwork model,
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
