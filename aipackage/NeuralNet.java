package aipackage;

import java.util.Iterator;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

/**
 * @author Peter Galvin
 * Neural Net: Main representation of a Neural Net
 */

public class NeuralNet {
    /*
     * layers: array of the number of nodes in each layer (input and output are also layers)
     * 
     * all indices start from 0
     */
    public NeuralNet(int [] layers) throws RuntimeException
    {
        if (layers.length < 2)
        {
            throw new RuntimeException("The NeuralNet must have at least two layers.");
        }
        m_layers = new ArrayList<List<Node>>(layers.length);
        
        for (int i = 0; i < layers.length; ++i)
        {
            List<Node> layer = new ArrayList<Node>(layers[i]);
            for (int k = 0; k < layers[i]; ++k)
            {
                layer.add(new Node(i, k, false));
            }
            m_layers.add(layer);
        }
    }
    
    /*
     * fully connect all the nodes on each layer and give each a threshold
     */
    public void connectAll()
    {
        Random generator = new Random();
        Iterator<List<Node>> iter = m_layers.iterator();
        
        // Connect each node to every node in a neighboring layer
        List<Node> pre_layer = iter.next();
        while (iter.hasNext()) {
            List<Node> cur_layer = iter.next();

            for (int i = 0; i < pre_layer.size(); ++i) {
                for (int j = 0; j < cur_layer.size(); ++j) {
                    addConnection(pre_layer, i, cur_layer, j, generator.nextDouble());
                }
            }
            for (Node node: cur_layer) {
                addThreshold(node, generator.nextDouble());
            }
            pre_layer = cur_layer;
        }
    }
    
    public void connectTest()
    {
        addConnection(0, 0, 1, 0, 0.2);
        addConnection(0, 1, 1, 0, 0.3);
        addConnection(0, 2, 1, 0, 0.4);
        addConnection(0, 3, 1, 1, 0.6);
        addConnection(0, 4, 1, 1, 0.7);
        addConnection(0, 5, 1, 1, 0.8);
        
        addConnection(1, 0, 2, 0, 1.0);
        addConnection(1, 1, 2, 0, 1.1);
        
        addThreshold(1, 0, 0.1);
        addThreshold(1, 1, 0.5);
        addThreshold(2, 0, 0.9);
    }
    
    void addConnection(int from_layer, int from_pos, int to_layer, int to_pos, double weight) {
        List<Node> layer_f = m_layers.get(from_layer);
        List<Node> layer_t = m_layers.get(to_layer);
        addConnection(layer_f, from_pos, layer_t, to_pos, weight);
    }
    
    void addConnection(List<Node> layer_f, int from_pos, List<Node> layer_t, int to_pos, double weight) {
        Node from_node = layer_f.get(from_pos);
        Node to_node = layer_t.get(to_pos);
        addConnection(from_node, to_node, weight);
    }
    
    void addConnection(Node from_node, Node to_node, double weight) {
        Connection con = new Connection(from_node, to_node, weight);
        from_node.addOutputConnection(con);
        to_node.addInputConnection(con);
    }
    
    // add a threshold to certain node
    void addThreshold(int layer, int pos, double weight) {
        List<Node> layer_i = m_layers.get(layer);
        Node node = layer_i.get(pos);
        addThreshold(node, weight);
    }
    
    void addThreshold(Node node, double weight) {        
        Node thrd = new Node(node.getLayer(), node.getPos(), true);
        thrd.setOutput(-1);
        
        Connection con = new Connection(thrd, node, weight);
        
        thrd.addOutputConnection(con);
        node.addInputConnection(con);
    }
    
    // r: rate parameter
    public void train(double [][] inputvs, double [][] outputvs, double r) throws RuntimeException
    {
    	// For each vector of inputvs:
    	// - Use the inputs to calculate an output based on the current weights
    		// calcedOutput can be gotten by using the NeuralNet function
			//  calcOutput which will make each node calculate its own output and
			//  store it as m_output
    		// This should be done layer by layer (outputs become new inputs
    		//  automatically)
    	// - Compute Beta = outputvs - calcedOutput (for output layer only)
    	// - Back propagate the error to get the error for each node in network,
    	//   including for the threshold nodes (beta is a variable for each node)
    	// - Compute weight change for each weight based on the beta for the
    	//   connected nodes, but don't implement it; save this weight change
    	// Do this process for each input vector, then sum up the delta w's and
    	//  finally implement it. This is one round.
    	
    	// Loops over every input-output data pair
    	for (int dataNum = 0; dataNum < inputvs.length; ++dataNum) {
    		double [] input = inputvs[dataNum];
    		double [] output = outputvs[dataNum];
    		
    		// Set the input data as the outputs of the first (input) layer
    		for (int i = 0; i < input.length; ++i) {
    			m_layers.get(0).get(i).setOutput(input[i]);
    		}
    		
    		// Loop over each layer (except first) to use each set of outputs as the new inputs
    		for (int i = 1; i < m_layers.size(); ++i) {
    			this.calcOutput(m_layers.get(i));
    		}
    		
    		// Set beta for each node in the output layer
    		List<Node> lastLayer = m_layers.get(m_layers.size()-1);
    		double results;
    		for (int i = 0; i < lastLayer.size(); ++i) {
    			results = lastLayer.get(i).getOutput();
    			lastLayer.get(i).setBeta(output[i] - results);
    		}
    			
    		// Set beta for each node in the hidden layers, propagating backwards,
    		// set deltaw for each connection, and add deltaw to the aggregate
    		// of all inputs for this round
    		for (int i = m_layers.size()-1; i > 0; --i) {
    			for (int j = 0; j < m_layers.get(i).size(); ++j) {
    				Node node = m_layers.get(i).get(j);
    				
    				if (i < m_layers.size()-1) {
    					node.calcBeta();
    				}
    				for (Connection con: node.getInputConnection()) {
    					con.calcDeltaw(r);
    					con.addDeltawToAggregate();
    				}
    			}
    		}
    	}
    	
    	// Apply the aggregate of the deltaw's for one round
    	for (int i = m_layers.size()-1; i > 0; --i) {
			for (int j = 0; j < m_layers.get(i).size(); ++j) {
				Node node = m_layers.get(i).get(j);
				
				for (Connection con: node.getInputConnection()) {
					con.applyDeltawAggregate();
				}
			}
		}
    	
    }
        
    // This method shall change the input and output of each node.
    public double [] evaluate(double [] inputv) throws RuntimeException {
        if (inputv.length != m_layers.get(0).size()) {
            throw new RuntimeException("incompabile inputv");
        }
        
        Iterator<List<Node>> iter = m_layers.iterator();
        List<Node> layer = iter.next();
        
        // input layer
        int i = 0;
        for (Node node: layer) {
            node.setOutput(inputv[i]);
            ++i;
        }
        
        while (iter.hasNext()) {
            layer = iter.next();
            calcOutput(layer);
        }

        // copy result
        double [] output = new double [layer.size()];
        i = 0;
        for (Node node: layer) {
            output[i] = node.getOutput();
            ++i;
        }
        
        return output;
    }
    
    public double error(double [][] inputvs, double [][] outputvs) throws RuntimeException
    {
        if (inputvs.length != outputvs.length)
        {
            throw new RuntimeException("inputvs and outputvs are not of the same length");
        }
        
        double error = 0;
        
        for (int i = 0; i < inputvs.length; ++i) {
            if (outputvs[i].length != m_layers.get(m_layers.size() - 1).size()) {
                throw new RuntimeException("incompatible outputs");
            }
            double [] results = evaluate(inputvs[i]);
            for (int j = 0; j < results.length; ++j) {
                error += (results[j] - outputvs[i][j]) * (results[j] - outputvs[i][j]);
            }
        }
        
        error /= inputvs.length;
        error = Math.pow(error, 0.5);
        
        return error;
    }
        
    public double errorrate(double [][]inputvs3, double [][]outputvs3, double threeOptions) {
        double accu = 0;
        for (int i = 0; i < inputvs3.length; ++i) {
            double [] inputs3 = inputvs3[i];
            double [] results = evaluate(inputs3);
            double target = outputvs3[i][outputvs3[i].length - 1];
            double ret = results[results.length - 1];
            //System.out.println("target is " + target + ", ret is " + ret);
            
            double boundOne = 0.5 - 0.25*threeOptions;
            double boundTwo = 0.5 + 0.25*threeOptions;
            if (target < boundOne) {  // target is 0
                if (ret > boundOne) {  // ret != target
                    ++accu;
                }
            } else if (target >= boundTwo){  // target is 1.0
                if (ret < boundTwo) {  // ret != target
                    ++accu;
                }
            } else {  // target is 0.5 (only occurs if threeOptions is true)
            	if (ret < boundOne || ret > boundTwo) {  // ret != target
            		++accu;
            	}
            }
        }
        
        double rate = accu / inputvs3.length;
        System.out.println("error rate is " + accu + "/" + inputvs3.length + " = " + rate);
        return rate;
    }
    
    private void calcOutput(List<Node> layer) {
        for (Node node: layer) {
            node.calcOutput();
        }           
    }
    
    
    private List<List<Node> > m_layers;

}
