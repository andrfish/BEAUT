/**
# This file demonstrates my modifed deep
# neural fitted q-iteration algorithm
#
# Author: Andrew Fisher
**/

package algorithm;

import neural.*;

import java.io.IOException;
import java.util.List;

import exception.*;

public class mnfq_iteration
{
    public int min_train_perc = 20;

    public neural[][] networks = null;
	
	private int states = 0;
	private int actions = 0;
	
	private double[] outputs = null;
	
    /**
    # This method initalizes the MNFQ algorithm with the following arguments:
    # states         == a count of the number of states
    # actions        == a count of the number of actions
    # learningR      == the learning rate for the network. An optional
    #                   argument IFF you pass the networkF argument; if 
    #                   it is passed with networkF, it will override
    #                   whatever value it reads from the file
    # ONE of the following arguments:
    # networkL       == a 1D array of the layers for the network. It is
    #                   assumed that the layers have already been populated
    #                   with neurons and that the ordering of the elements
    #                   is such that the first element is the input layer
    #                   which is connected to the second to N elements for
    #                   the hidden layer, and the last layer is the output
    #                   layer. This argument has priority over networkMatrix.
    # networkM       == a 1D array where the element is the layer
    #                   and the value is the number of neurons in that layer.
    #                   It is assumed that the matrix is built such that the 
    #                   first element is the input layer which is connected 
    #                   to the second to N elements for the hidden layer, 
    #                   and the last layer is the output layer.
    # networkF       == a 2D set of file names that points to an export as defined in the
    #                   export_network method for this network to import
    # networkT       == an 2D set of arrays of text that contains the export data of
    #                   a network, as defined in the export_network method
    # And the following optional argument:
    # outputs        == the outputs to compare the networks' to. 
    #                   An optional array for each network but it must be set 
    #                   before propgating forward
    **/
    public mnfq_iteration(Integer states, Integer actions, Double learningR, List<layer> networkL, int[] networkM, String networkF, String[][][] networkT, double[] outputs) throws InvalidNeuronCount, NoLearningRate, InvalidLayerCount, IOException, InvalidLength, MissingArgument
    {
        //Check that the states and actions were passed
        if(states == null)
            throw new MissingArgument("The states have not been set");
        if(actions == null)
            throw new MissingArgument("The actions have not been set");

        //Check that the states and actions are valid
        if(states == 0)
            throw new InvalidLength("The amount of states must be greater than zero");
        if(actions == 0)
            throw new InvalidLength("The amount of states must be greater than zero");

    	//The number of states and actions should be equal
    	//The action performed is the new state
        if(states.intValue() != actions.intValue())
        {
    	    throw new InvalidLength("The number of states must equal the number of actions");
        }

        //Initialize a list of networks
        this.networks = new neural[states][actions];
        
        //Check if the layers were passed
        if(networkL != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a layer array");

            //Create (states by actions) networks
            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < actions; j++)
                    if(outputs != null)
                    {
                    	double[] temp = {outputs[j]};
                        this.networks[i][j] = new neural(learningR, null, temp, networkL, null, null, null, null);
                    }
                    else
                        this.networks[i][j] = new neural(learningR, null, null, networkL, null, null, null, null);
            }
        }
        //Else, check if we need to generate the network from the matrix
        else if(networkM != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a matrix");

            //Create (states by actions) networks
            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < actions; j++)
                {
                    if(outputs != null)
                    {
                    	double[] temp = {outputs[j]};
                        this.networks[i][j] = new neural(learningR, null, temp, null, networkM, null, null, null);
                    }
                    else
                        this.networks[i][j] = new neural(learningR, null, null, null, networkM, null, null, null);
                }
            }
        }
        //Else, import the file
        else if (networkF != null)
        {
            //Create (states by actions) networks
            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < actions; j++)
                {
                    if(outputs != null)
                    {
                    	double[] temp = {outputs[j]};
                        this.networks[i][j] = new neural(learningR, null, temp, null, null, networkF, null, null);
                    }
                    else
                        this.networks[i][j] = new neural(learningR, null, null, null, null, networkF, null, null);
                }
            }
        }
        //Else, import from array
        else
        {
            //Create (states by actions) networks
            for (int i = 0; i < states; i++)
            {
                for (int j = 0; j < actions; j++)
                {
                    if(outputs != null)
                    {
                    	double[] temp = {outputs[j]};
                        this.networks[i][j] = new neural(learningR, null, temp, null, null, null, networkT[i][j], null);
                    }
                    else
                        this.networks[i][j] = new neural(learningR, null, null, null, null, null, networkT[i][j], null);
                }
            }
        }

        //Store the passed values
        this.states = states;
        this.actions = actions;
        this.outputs = outputs;
    }

    public double[] calc_immediate_offset(int i, int j, double[][] table, int curPop, int weeksLeft, double curQ, boolean learn) throws OutputNotSet, NoConnectionException, NotEqualLength, InvalidNeuronCount
    {
        //Cycle each q-value through its network
        //This will store the lowest absolute error and the corresponding q-value
        double lowestError = Double.MIN_VALUE;
        double alphaQ = curQ;
        
        double [] inp = {curPop, weeksLeft, 0}; //NOTATION: weeksLeft == \tau
        double[] output = {this.outputs[j]};
        
        neural curNetwork = this.networks[i][j];
        
        //Put the alpha q through the network and back propagate
        inp[2] = alphaQ;
        curNetwork.set_output(output);
        curNetwork.set_input(inp);
        curNetwork.propagate_forward(true);
        
	   double networkOutput = curNetwork.get_network_output()[0]; //NOTATION: \hat{n}_{i, j, p, t}

        if (networkOutput!=networkOutput){
            networkOutput = 0.0;
        }
        
        // BartG
	    lowestError = this.outputs[j] - networkOutput; //NOTATION: \E^*_{i, j, p, t}

        //If the error is less than 20%, back propagate
        double avg = (this.outputs[j] + networkOutput) / 2.0;
        double perError;
        if (networkOutput==networkOutput){
            perError= 100 * (Math.abs(lowestError) / avg);
        } else {
            perError = 100.0;
        }
        
        // System.out.println("perce check: " + Math.abs(perError) +" < "+ this.min_train_perc);
        if(Math.abs(perError) < this.min_train_perc && learn)
        {
            curNetwork.propagate_backward(true);
        }
        else
        {
        	curNetwork.previousErrors.remove(curNetwork.previousErrors.size() - 1);
        }
        
        //Sum up the q-values (with alpha instead of the value at that state)
        double summ = 0;
        for (int k = 0; k < table.length; k++)
        {
            if(k != i)
                summ += table[k][j];
            else
                summ += alphaQ;
        }
        
        double ret = alphaQ / summ;

        try
        {
            ret *= (-lowestError)/this.outputs[j]; //NOTATION: r^*_{p, t}
        }
        catch (Exception e)// Output is likely zero
        {
            ret *= (-lowestError);
        }
        
        double[] temp = new double[2];
        temp[0] = ret;
        temp[1] = perError;
        return temp;
    }


    public double calc_offset(int i, int j, double[][] table, double[] qValues, int curPop, int weeksLeft) throws OutputNotSet, NoConnectionException, InvalidNeuronCount, NotEqualLength
    {
        //Cycle each q-value through its network
        //This will store the lowest absolute error and the corresponding q-value
        double lowestError = Double.MAX_VALUE;
        double alphaQ = Double.MAX_VALUE;
        
        double[] inp = {curPop, weeksLeft, 0};
        double[] output = {this.outputs[j]};
        
        neural curNetwork = this.networks[i][j];
        for (double q : qValues)
        {
            //Set the q-value input
        	//q = qValues[qi];
            inp[2] = q;
            
            //Set the network's input and output
            curNetwork.set_output(output);
            curNetwork.set_input(inp);
            
            //Propagate forward
            curNetwork.propagate_forward(true);
            
            //Determine the relative error
            double curOutput = curNetwork.get_network_output()[0];
            double error = this.outputs[j] - curOutput;
            //errors[qi] = error;
            //See if the absolute error is the lowest
            if(Math.abs(error) < Math.abs(lowestError))
            // if(Math.abs(error) > Math.abs(lowestError))
            {
                lowestError = error;
                alphaQ = q; //NOTATION: m
            }
        }
        
        //Put the alpha q through the network and back propagate
        inp[2] = alphaQ;
        curNetwork.set_output(output);
        curNetwork.set_input(inp);
        curNetwork.propagate_forward(true);
        curNetwork.propagate_backward(true);
        
        //Sum up the q-values (with alpha instead of the value at that state)
        double summ = 0;
        for (int k = 0; k < table.length; k++)
        {
            if(k != i)
                summ += table[k][j];
            else
                summ += alphaQ;
        }
        
        double ret = alphaQ / summ;
        
        try
        {
            ret *= (-lowestError)/this.outputs[j];
        }
        catch (Exception e) //Output is likely zero
        {
            ret *= (-lowestError);
        }

        return ret; //NOTATION: \delta Q_{i, j, t}
    }

    /**
    # This internal method sets the output of the MNFQ algorithm
    **/
    public void _set_output(double[] output)
    {
        this.outputs = output;
    }

    /**
    # This internal method sets the learning rate of the MNFQ algorithm
    **/
    public void _set_learning_rate(double lr)
    {
        for (int i = 0; i < this.states; i++)
        {
            for (int j = 0; j < this.actions; j++)
                this.networks[i][j].set_learning_rate(lr);
        }
    }
    
    public double _get_learning_rate()
    {
    	return this.networks[0][0].get_learning_rate();
    }
    
    public double cross_entropy(double yhat, double y){
	    if (y == 1.0){
	        return -1.0 * Math.log(yhat);
	    } else {
	        return -1.0 * Math.log(1.0 - yhat);
	    }
    }
    
    public void _adjust_learning_rate_decay(int epoch, double decay){
        for (int i = 0; i < this.states; i++)
        {
            for (int j = 0; j < this.actions; j++)
                this.networks[i][j].adjust_learning_rate_decay(epoch, decay);
        }
    }

    public void _adjust_learning_rate_sched(int epoch, double decay, int patience){
        for (int i = 0; i < this.states; i++)
        {
            for (int j = 0; j < this.actions; j++)
                this.networks[i][j].adjust_learning_rate_sched(epoch, decay, patience);
        }
    }
    
    public int _adjust_random_restart(int epoch, int patience, boolean lastLayerOnly)
    {
    	int reset_count = 0;
    	
        for (int i = 0; i < this.states; i++)
        {
            for (int j = 0; j < this.actions; j++)
            {
                boolean restarted = this.networks[i][j].adjust_random_restart(epoch, this.networks[i][j].previousErrors, patience, lastLayerOnly);
                
                if (restarted)
                	reset_count += 1;
            }
        }
    	
    	return reset_count;
    }
    
    public void _reset_errors()
    {
        for (int i = 0; i < this.states; i++)
        {
            for (int j = 0; j < this.actions; j++)
            {
                this.networks[i][j].previousErrors.clear();
            }
        }
    }
}
