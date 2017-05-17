import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Created by breandanconsidine on 5/16/17.
 */
public class Trainer {
    private static final Logger log = LoggerFactory.getLogger(Logger.class);
    static int outputNum = 10; // The number of possible outcomes
    static int numEpochs = 10; // Number of training epochs

    static void printStats(DataSetIterator mnistTest, MultiLayerNetwork net) {
        //Perform evaluation (distributed)
        Evaluation evaluation = net.evaluate(mnistTest);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        log.info("***** Example Complete *****");
    }


    static void train(MultiLayerNetwork model,
                      DataSetIterator train,
                      DataSetIterator test,
                      String saveAsFilename) throws IOException {
        //Save the model
        File locationToSave = new File("javaday-examples/src/main/resources/" + saveAsFilename);
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
