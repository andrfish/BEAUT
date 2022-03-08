/**
# This file represents an adaptable parameter which
# will output values between 0 to 1
#
# Author: Andrew Fisher
**/

package algorithm;

import java.io.IOException;

import exception.*;
import neural.neural;
import neural.activation;

public class adaptable_param
{
	private neural network = null;
	private String activationFunc = null;

    /**
    # This method initalizes the parameter with the following arguments:
    # inputs         == the number of inputs for the parameter
    # outputs        == the number of outputs for the parameter
    #
    # And the following optional arguments:
    # activationFnc  == the function to apply to each of the output
    #                   neurons. By default, the "exploding" sigmoid 
    #                   function is used but None could be passed to 
    #                   simply use the network's output if desired
    # learningRate   == the learning rate for the parameter
    # seed           == a seed for the random generator
    # networkM       == a 1D array where the element is the layer
    #                   and the value is the number of neurons in that layer.
    #                   It is assumed that the matrix is built such that the 
    #                   first element is the input layer which is connected 
    #                   to the second to N elements for the hidden layer, 
    #                   and the last layer is the output layer.
    **/
    public adaptable_param(int inputs, int outputs, String activationFunc, double learningRate, Integer seed, int[] networkM) throws InvalidNeuronCount, NoLearningRate, InvalidLayerCount, IOException
    {
        //Create the network
        if(networkM == null)
        {
            networkM = new int[3];

            //Define the input layer
            networkM[0] = inputs;

            //Define the hidden layer
            networkM[1] = (inputs + outputs) * 2;
            
            //Define the output layer
            networkM[2] = outputs;
        }
        
        //(Double learningR, double[] inputA, double[] outputA, List<layer> networkL, int[] networkM, String networkF, String[] networkT, Integer seed)
        this.network = new neural(learningRate, null, null, null, networkM, null, null, seed);
        this.activationFunc = activationFunc.toLowerCase();
    }

    /**
    # This method sets the output of the network which is needed 
    # if training
    **/
    public void set_output(double[] val) throws InvalidNeuronCount
    {
        this.network.set_output(val);
    }

    /**
    # This method sets the input of the network which is needed
    # if propogating forward
    **/
    public void set_input(double[] val) throws InvalidNeuronCount
    {
        this.network.set_input(val);
    }

    /**
    # This method gets the output for the network
    **/
    public double[] val()
    {
        double[] ret = this.network.get_network_output();

        //Apply the activation function
        if(this.activationFunc != null)
        {
            for (int i = 0; i < ret.length; i++)
            {
            	if (this.activationFunc.equals("ex_sigmoid"))
            		ret[i] = activation.ex_sigmoid(ret[i]);
            	else if (this.activationFunc.equals("tanh"))
            		ret[i] = activation.tanh(ret[i]);
            	else
            		ret[i] = activation.sigmoid(ret[i]);
            }
        }

        //Ensure this value is < 1 and >= 0
        for (int i = 0; i < ret.length; i++)
        {
            double val = Math.abs(ret[i]);
            while(val >= 1)
                val /= 10;
            ret[i] = val;
        }

        return ret;
    }

    /**
    # This method propogates the input through the network and
    # returns the output.
    **/
    public Object[] process_input(boolean train) throws OutputNotSet, NoConnectionException, InvalidLength, NotEqualLength
    {
        this.network.propagate_forward(true);

        //Get the processed output
        double[] output = this.val();
        this.network._set_output(output);

        //Set the error
        double[] actual = this.network.output;
        double[] error = new double[output.length];
        for (int i = 0; i < output.length; i++)
            error[i] = actual[i] - output[i];
        this.network._set_output_error(error);

        if(train)
            this.network.propagate_backward(train);

        Object[] ret = {output, error};
        return ret;
    }
}