/**
# This file represents workers used in my neural
# network code. This class should ONLY be used
# internally.
#
# Author: Andrew Fisher
**/

package neural;

import java.util.ArrayList;
import java.util.List;

import exception.*;

public class workers
{
	/**
	# Calculate the output for each neuron in the next layer
	# This method should NOT be called manually. It is for internal use
	# only.
	**/
	public static double _forward_worker(neuron neuron, List<neuron> neurons, int i, int count)
	{
	    double output = 0;
	    for (int j = 0; j < count; j++)
	        output += neurons.get(j).get_output() * neurons.get(j).weightList[i];
	    
	    //Add the bias if there is one
	    if (neuron.get_bias() != null)
	        output += neuron.get_bias().doubleValue();
	    
	    //Return the output
	    return output;
	}
	
	/**
	# Calculate the error for each neuron and update their weights
	# This method should NOT be called manually. It is for internal use
	# only.
	**/
	@SuppressWarnings("removal")
	public static List<double[]> _backward_worker(neuron neuron, List<neuron> neurons, boolean learn, int count, double learningRate) throws OutputNotSet
	{
	    double error = 0;
	    double[] weights = neuron.get_weights();
	
	    //Cycle through each connection's neuron
	    for (int j = 0; j < count; j++)
	    {
	        Double nextOutput = neurons.get(j).get_output();
	
	        //Ensure that the output has been set
	        if (nextOutput == null)
	            throw new OutputNotSet("The output for this neuron is not set (have you propagated forward yet?)");
	
	        double nextError = neurons.get(j).get_error().doubleValue();
	        double activationVal = activation.sigmoid(nextOutput);
	
	        //Add on to the error
	        error += nextError * activationVal * neuron.get_weight(j);
	        
	        //Check if it should learn
	        if(learn)
	        {
	            //Update the weight
	            weights[j] += (learningRate * nextError * activationVal * neuron.get_output().doubleValue());
	        }
       }
	        
	    //Return the output
	    List<double[]> ret = new ArrayList<double[]>();
	    List<Double> temp = new ArrayList<Double>();
	    temp.add(new Double(error));
	    ret.add(temp.stream().mapToDouble(d -> d.doubleValue()).toArray());
	    ret.add(weights);
	    
	    return ret;
	}
}