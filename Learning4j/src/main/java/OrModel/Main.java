package OrModel;

import java.util.List;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

@SuppressWarnings("deprecation")
public class Main {

	public static void main(String[] args) {
		
		INDArray trainingSet = Nd4j.zeros(4,2);
		trainingSet.putScalar(0, 0, 0);
		trainingSet.putScalar(0, 1, 0);
		trainingSet.putScalar(1, 0, 0);
		trainingSet.putScalar(1, 1, 1);
		trainingSet.putScalar(2, 0, 1);
		trainingSet.putScalar(2, 1, 0);
		trainingSet.putScalar(3, 0, 1);
		trainingSet.putScalar(3, 1, 1);
		INDArray resultSet = Nd4j.zeros(4,1);
		resultSet.putScalar(0, 0);
		resultSet.putScalar(1, 1);
		resultSet.putScalar(2, 1);
		resultSet.putScalar(3, 1);
		
		DataSet dataSet = new DataSet(trainingSet, resultSet);
		List<DataSet> listDataSets = dataSet.asList();
		int batchSize = 1;
		DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSets,batchSize);
		
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(1e-4))
				.miniBatch(false) //Since really small training data
				.list()
				.layer(new DenseLayer.Builder()
						.nIn(2)
						.nOut(4)
						.activation(Activation.RELU)
						.weightInit(new UniformDistribution(0, 1)) // Init weights to a value between 0 and 1
						.build())
				.layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
						.nIn(4)
						.nOut(1)
						.activation(Activation.SIGMOID)
						.weightInit(new UniformDistribution(0, 1)).build())
				.build();

		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		
		model.init();
		//print the score with every 500 iteration
		model.setListeners(new ScoreIterationListener(500));
		model.fit(dsi , 1000); // 1000 represents epochs
		
		
		INDArray test1 = Nd4j.zeros(1,2);
		test1.putScalar(0, 0, 0);
		test1.putScalar(0, 1, 0);
		INDArray test2 = Nd4j.zeros(1,2);
		test2.putScalar(0, 0, 0);
		test2.putScalar(0, 1, 1);
		INDArray test3 = Nd4j.zeros(1,2);
		test3.putScalar(0, 0, 1);
		test3.putScalar(0, 1, 0);
		INDArray test4 = Nd4j.zeros(1,2);
		test4.putScalar(0, 0, 1);
		test4.putScalar(0, 1, 1);
		
		System.out.println("Input (0,0) Output: " + model.output(test1));
		System.out.println("Input (0,1) Output: " + model.output(test2));
		System.out.println("Input (1,0) Output: " + model.output(test3));
		System.out.println("Input (1,1) Output: " + model.output(test4));
		
		Evaluation eval = model.evaluate(dsi);
		System.out.println(eval.stats());
		System.out.print(model.summary());

	}

}
