/**
# This file represents a layer in a neural network
#
# You can build the network automatically using the
# neural.py implementation instead of using this
# class directly
#
# Author: Andrew Fisher
**/

package neural;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import exception.*;

public class layer
{
	private Random rnd = null;
	private List<neuron> neuronList = null;
	
	protected int count = 0;
	private layer nextLayer = null;
	
	protected double learningRate = 0.0;
	
    /**
    # This method initalizes the layer with the following arguments:
    # size    == the number of neurons in the layer
    # learningR == the learning rate for this layer
    # And the following optional argument:
    # neuronL == the list of neurons in this layer. Size must be equal
    #            to the length of this list. The argument is optional, 
    #            if none is specified, a list of them will be randomly
    #            created based on the size.
    # nextL   == the next layer connected to this layer. The argument is
    #            optional and if it is not passed, you are saying that this 
    #            is the output layer.
    # seed    == a seed for the random generator
    **/
    public layer(int size, double learningR, List<neuron> neuronL, layer nextL, Integer seed) throws InvalidLayerCount
    {
        //Error checking for the size
        if (size <= 0)
            throw new InvalidLayerCount("The layer size must be a positive number (" + size + ")");

        //Set the seed if needed
        if(seed != null)
        {
            this.rnd = new Random(seed);
        }
        else
        {
            this.rnd = new Random();
        }

        //If a neuron list was passed, check that the size is equal to it
        if(neuronL != null)
        {
            if(size != neuronL.size())
                throw new InvalidLayerCount("The layer size (" + size + ") must be equal to the neuron list length (" + neuronL.size() + ")");
            else
                this.neuronList = neuronL;
        }
        //Else, need to create the list of neurons
        else
        {
            this.neuronList = new ArrayList<neuron>();
            for (int _i = 0; _i < size; _i++)
            {
                //First, randomly create weights, if needed, in the range -0.5 to 0.5
                double[] weights = null;
                if(nextL != null)
                {
                    weights = new double[nextL.count];
                    for (int _j = 0; _j < nextL.count; _j++)
                        weights[_j] = -0.5 + (0.5 - (-0.5)) * rnd.nextDouble();
                }

                //Next, create the neuron with a bias of 1.0
                this.neuronList.add(new neuron(weights, 1.0));
            }
        }
        
        this.count = size;

        if(nextL != null)
            this.nextLayer = nextL;
        else
            this.nextLayer = null;

        //The code assumes that this will be the same for all layers
        this.learningRate = learningR;
    }

    /**
    # This method propagates forward in the network from this layer
    **/
    public void forward() throws NoConnectionException
    {
        //Check that there is a layer to propagate forward to
        if(this.get_next_layer() == null)
            throw new NoConnectionException("There is no layer to propagate forward to");

        List<neuron> neurons = this.get_neuron_list();
        List<neuron> nextNeurons = this.get_next_layer().get_neuron_list();
        for (int i = 0; i < this.get_next_layer().get_count(); i++)
        {
            //Set the output
            neuron neuron = nextNeurons.get(i);
            neuron.set_output(workers._forward_worker(neuron, neurons, i, this.get_count()));
        }
    }

    /**
    # This method propagates backward in the network from this layer
    **/
    public void backward(boolean learn) throws OutputNotSet, NoConnectionException
    {
        //Check that there is a layer to propagate backward from
        if(this.get_next_layer() == null)
            throw new NoConnectionException("There is no layer to propagate backward from");

        List<neuron> neurons = this.get_neuron_list();
        List<neuron> nextNeurons = this.get_next_layer().get_neuron_list();
        for (int i = 0; i < this.get_count(); i++)
        {
            //Set the error and weights if applicable
            neuron neuron = neurons.get(i);
            List<double[]> ret = workers._backward_worker(neuron, nextNeurons, learn, this.get_next_layer().get_count(), this.learningRate);
            neuron.set_error(ret.get(0)[0]);

            if(learn)
                neuron.set_weights(ret.get(1));
        }
        
        //Update the biases for the next layer if applicable
        if(learn)
        {
            for (int i = 0; i < nextNeurons.size(); i++)
            {
                neuron curNeuron = nextNeurons.get(i);
                Double curBias = curNeuron.get_bias();
                if(curBias != null)
                {
                    double newBias = curBias.doubleValue() + (this.learningRate * curNeuron.get_error() * activation.sigmoid(curNeuron.get_output().doubleValue()));
                    curNeuron.set_bias(newBias);
                }
            }
        }
    }

    /**
    # This method sets the next layer for this layer and randomly
    # generates the weights to it from this layer's neurons
    # You can pass false to not regen the weights but this should only
    # be done if you have already set the weights for the neurons and 
    # are sure it is the correct number of connections needed
    **/
    public void set_next_layer(layer nextL, boolean regenWeights)
    {
        this.nextLayer = nextL;

        if (regenWeights)
        {
            for (int i = 0; i < this.count; i++)
            {
                    //Randomly create weights in the range -0.5 to 0.5
                    double[] weights = new double[this.nextLayer.count];
                    for (int _j = 0; _j < this.nextLayer.count; _j++)
                        weights[_j] = -0.5 + (0.5 - (-0.5)) * rnd.nextDouble();

                    //Assign weights to the neuron
                    this.neuronList.get(i).set_weights(weights);
            }
        }
    }
    
    public void randomize_weights()
    {
    	if (this.nextLayer != null)
    	{
    		this.set_next_layer(this.nextLayer, true);
    	}
    }

    /**
    # This method returns the neuron list for this layer
    **/
    public List<neuron> get_neuron_list()
    {
        return this.neuronList;
    }

    /**
    # This method returns the count for this layer
    **/
    public int get_count()
    {
        return this.count;
    }
    
    /**
    # This method returns the next layer for this layer
    **/
    public layer get_next_layer()
    {
        return this.nextLayer;
    }
}