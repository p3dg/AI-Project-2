package aipackage;

import java.io.FileNotFoundException;
import java.util.Random;

/**
 * @author Peter Galvin
 * Neural Net Learner is the main class for going through the data and learning
 * the Lenses Data provided by CMU.
 */
public class NeuralNetLearner {
    /**
     * @param args
     * @throws FileNotFoundException
     */
    public static void main(String[] args) throws FileNotFoundException {
        int[] layers = { 6, 2, 1 }; // three layers
        NeuralNet net = new NeuralNet(layers);
        net.connectTest();

        double[][] inputvs = { { 1, 1, 0, 0, 0, 0 }, { 1, 0, 1, 0, 0, 0 },
                { 1, 0, 0, 1, 0, 0 }, { 1, 0, 0, 0, 1, 0 },
                { 1, 0, 0, 0, 0, 1 }, { 0, 1, 1, 0, 0, 0 },
                { 0, 1, 0, 1, 0, 0 }, { 0, 1, 0, 0, 1, 0 },
                { 0, 1, 0, 0, 0, 1 }, { 0, 0, 1, 1, 0, 0 },
                { 0, 0, 1, 0, 1, 0 }, { 0, 0, 1, 0, 0, 1 },
                { 0, 0, 0, 1, 1, 0 }, { 0, 0, 0, 1, 0, 1 },
                { 0, 0, 0, 0, 1, 1 } };

        double[][] outputvs = { { 0 }, { 0 }, { 1 }, { 1 }, { 1 }, { 0 },
                { 1 }, { 1 }, { 1 }, { 1 }, { 1 }, { 1 }, { 0 }, { 0 }, { 0 } };

        double error = 0;
        
        System.out.println("Demo 1:");
        
        for (int n = 0; n < 300; ++n) {
            net.train(inputvs, outputvs, 1);
            
            if ((n+1)%100 == 0) {
            	error = net.error(inputvs, outputvs);
            	System.out.println("error is " + error);
            }
        }

        net.errorrate(inputvs, outputvs, 0);
        System.out.println("============================");

        int[] layers2 = { 2, 2 }; // two layers
        NeuralNet net2 = new NeuralNet(layers2);
        net2.connectAll();

        double[][] inputvs2 = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 1, 0 } };
        double[][] outputvs2 = { { 0, 0 }, { 0, 1 }, { 1, 1 }, { 0, 1 } };

        error = 0;

        System.out.println("Demo 2:");
        
        for (int n = 0; n < 300; ++n) {
            net2.train(inputvs2, outputvs2, 1);
            
            if ((n+1)%100 == 0) {
            	error = net2.error(inputvs2, outputvs2);
            	System.out.println("error is " + error);
            }
        }
        
        net2.errorrate(inputvs2, outputvs2, 0);
        System.out.println("============================");

        DataProcessor data = new DataProcessor("../crx.data.training");
        int[] layers3 = { 15, 30, 1 }; // two layers
        NeuralNet net3 = new NeuralNet(layers3);
        net3.connectAll();
        
        double[][] inputvs3 = data.m_inputvs;
        double[][] outputvs3 = data.m_outputvs;
        
        data = new DataProcessor("../crx.data.testing");
        double[][] testInput3 = data.m_inputvs;
        double[][] testOutput3 = data.m_outputvs;
        
        error = 0;
        
        for (int n = 0; n < 300; ++n) {
            net3.train(inputvs3, outputvs3, 0.1);
            
            if (n%20 == 0) {
                error = net3.error(inputvs3, outputvs3);
            }
            // Throw away any neural net that doesn't meet this early error
            // threshold, to ensure using only the best possible neural nets
            if (error > 0.31 && n >= 20) {
            	net3 = new NeuralNet(layers3);
                net3.connectAll();
                n = 0;
                error = 0;
            }
        }
        
        System.out.println("Credit Data:");
        
        System.out.print("Training ");
        error = net3.error(inputvs3, outputvs3);
    	System.out.println("error is " + error);
        
        System.out.print("Training ");
        net3.errorrate(inputvs3, outputvs3, 0);
        
        System.out.print("Testing ");
        error = net3.error(testInput3, testOutput3);
    	System.out.println("error is " + error);

        System.out.print("Testing ");
        net3.errorrate(testInput3, testOutput3, 0);
        System.out.println("============================");

        int[] layers4 = { 4, 8, 1 }; // two layers
        NeuralNet net4 = new NeuralNet(layers4);
        net4.connectAll();

        // The raw Lenses data
        double[][] inputvs4 = { { 1, 1, 1, 1 }, { 1, 1, 1, 2 }, { 1, 2, 1, 1 },
        						{ 1, 2, 1, 2 }, { 1, 2, 2, 1 }, { 1, 2, 2, 2 },
        						{ 2, 1, 1, 1 }, { 2, 1, 1, 2 }, { 2, 1, 2, 2 },
        						{ 2, 2, 1, 1 }, { 2, 2, 2, 1 }, { 2, 2, 2, 2 },
        						{ 3, 1, 1, 2 }, { 3, 1, 2, 1 }, { 3, 1, 2, 2 }, 
        						{ 3, 2, 1, 1 }, { 3, 2, 2, 1 }, { 3, 2, 2, 2 } };
        double[][] outputvs4 = { { 3 }, { 2 }, { 3 }, { 2 }, { 3 }, { 1 },
        						 { 3 }, { 2 }, { 1 }, { 3 }, { 3 }, { 3 },
        						 { 3 }, { 3 }, { 1 }, { 3 }, { 3 }, { 3 } };
        
        double[][] testInput4 = { { 1, 1, 2, 1 }, { 1, 1, 2, 2 },
        						  { 2, 1, 2, 1 }, { 2, 2, 1, 2 },
        						  { 3, 1, 1, 1 }, { 3, 2, 1, 2 } };
        double[][] testOutput4 = { { 3 }, { 1 }, { 3 }, { 2 }, { 3 }, { 2 } };
        
        // Normalization
        // Input training data:
        double[] columnTotal = new double[4];
        for(int j = 0; j < 4; j++) {
        	// Compute columnTotal
        	for(int k = 0; k < 18; k++) {
                columnTotal[j] += inputvs4[k][j];
            }
        	
            // Compute average and stddev
            double ave = columnTotal[j]/18;
            
            double sumSquareDifs = 0;
            for(int k = 0; k < 18; k++) {
                sumSquareDifs += Math.pow((inputvs4[k][j] - ave), 2);
            }
            double stddev = Math.sqrt(sumSquareDifs/18);
            
            for(int k = 0; k < 18; k++) {
                inputvs4[k][j] = (inputvs4[k][j]-ave)/stddev;
            }
        }
        // Input testing data:
        columnTotal = new double[4];
        for(int j = 0; j < 4; j++) {
        	// Compute columnTotal
        	for(int k = 0; k < 6; k++) {
                columnTotal[j] += testInput4[k][j];
            }
        	
            // Compute average and stddev
            double ave = columnTotal[j]/6;
            
            double sumSquareDifs = 0;
            for(int k = 0; k < 6; k++) {
                sumSquareDifs += Math.pow((testInput4[k][j] - ave), 2);
            }
            double stddev = Math.sqrt(sumSquareDifs/6);
            
            for(int k = 0; k < 6; k++) {
                testInput4[k][j] = (testInput4[k][j]-ave)/stddev;
            }
        }
        // Output:
        for (int i = 0; i < 18; ++i) {
        	outputvs4[i][0] = (outputvs4[i][0] - 1) / 2.0;
        }
        for (int i = 0; i < 6; ++i) {
        	testOutput4[i][0] = (testOutput4[i][0] - 1) / 2.0;
        }
        
        error = 0;
        
        // Train and print error
        for (int n = 0; n < 10000; ++n) {
            net4.train(inputvs4, outputvs4, 2);
        }
        
        System.out.println("Lenses Data:");
        
        System.out.print("Training ");
        error = net4.error(inputvs4, outputvs4);
    	System.out.println("error is " + error);
        
        System.out.print("Training ");
        net4.errorrate(inputvs4, outputvs4, 1);
        
        System.out.print("Testing ");
        error = net4.error(testInput4, testOutput4);
    	System.out.println("error is " + error);

        System.out.print("Testing ");
        net4.errorrate(testInput4, testOutput4, 1);
        
        return;
    }
}
