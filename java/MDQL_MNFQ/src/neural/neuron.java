/**
# This file represents a neuron in a layer in a 
# neural network
#
# You can build the network automatically using the
# neural.py implementation instead of using this
# class directly
#
# Author: Andrew Fisher
**/
package neural;

public class neuron
{
	double[] weightList = null;
	Double bias = null;
	
	Double output = null;
	Double error = null;
	
	/**
	# This method initalizes the layer with the following arguments:
	# weightL == the list of weights to the next layer if applicable.
	#            it is assumed that there is a 1-to-1 relationship with
	#            this list and the next layer. That is, the first element
	#            in the list is the weight to the first neuron in the 
	#            next layer and so forth. The argument is optional and 
	#            if it is not passed, you are saying that it is an output
	#            neuron.
	# biasV   == the bias for this neuron. The argument is optional.
	**/
	public neuron(double[] weightL, Double biasV)
	{
	    this.weightList = weightL;
	
	    if(biasV != null)
	        this.bias = biasV;
	    else
	        this.bias = null;
	
	    this.output = null;
	    this.error = null;
	}
	
	/**
	# This method sets a weight for the neuron
	**/
	public void set_weight(int i, double weight)
	{
	    this.weightList[i] = weight;
	}
	
	/**
	# This method sets all of the weights for the neuron
	**/
	public void set_weights(double[] weights)
	{
	    this.weightList = weights;
	}
	
	/**
	# This method sets the output for the neuron
	**/
	@SuppressWarnings("removal")
	public void set_output(double outputV)
	{
	    this.output = new Double(outputV);
	}
	
	/**
	# This method sets the bias for the neuron
	**/
	@SuppressWarnings("removal")
	public void set_bias(double biasV)
	{
	    this.bias = new Double(biasV);
	}
	
	/**
	# This method sets the error for the neuron
	**/
	@SuppressWarnings("removal")
	public void set_error(double errorV)
	{
	    this.error = new Double(errorV);
	}
	
	/**
	# This method returns a weight
	**/
	public double get_weight(int i)
	{
	    return this.weightList[i];
	}
	
	/**
	# This method returns all weights
	**/
	public double[] get_weights()
	{
	    return this.weightList;
	}
	
	/**
	# This method returns the output for this neuron
	**/
	public Double get_output()
	{
		return (this.output == null ? null : this.output);
	}
	
	/**
	# This method returns the bias for this neuron
	**/
	public Double get_bias()
	{
		return (this.bias == null ? null : this.bias);
	}
	
	/**
	# This method returns the error for this neuron
	**/
	public Double get_error()
	{
		return (this.error == null ? null : this.error);
	}
}