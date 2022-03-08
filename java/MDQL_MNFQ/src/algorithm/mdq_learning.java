/**
# This file demonstrates my modified deep
# q-learning algorithm
#
# Author: Andrew Fisher
**/

package algorithm;

import neural.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.IntStream;
import java.util.Arrays;

import exception.*;

public class mdq_learning
{
    public double loss_hold = 0.0;

    double [] mnfq_outputs = null;
    
    public String use_error_adjust;
    public String use_error_metric;
    
    public int min_train_perc;
    public boolean weighted_err;

    public neural network = null;
	
	public String[] states = null; 	//NOTATION: S
	public String[] actions = null; //NOTATION: Assumed to also be S; it's defining the state to transition to
	
	private int[][] impossible = null;
	
	private boolean onehot = false;
	
	private Random rnd = null;
	
	public List<double[][]> tables = null;
	
	private boolean random_table = false;
	
	public mnfq_iteration mnfq = null;
	
    /**
    # This method initalizes the MDQ network with the following arguments:
    # states         == a list of the possible states in the dataset
    # actions        == a list of the possible actions in the dataset
    # learningR      == the learning rate for the algorithm
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
    # networkF       == a file name that points to an export as define in the
    #                   export_network method for this network to import
    # networkT       == an array of text that contains the export data of
    #                   a network, as defined in the export_network method
    # Plus ONE of the following arguments for the MNFQ networks:
    # mnfqL          == a 1D array of the layers for the network. It is
    #                   assumed that the layers have already been populated
    #                   with neurons and that the ordering of the elements
    #                   is such that the first element is the input layer
    #                   which is connected to the second to N elements for
    #                   the hidden layer, and the last layer is the output
    #                   layer. This argument has priority over networkMatrix.
    # mnfqM          == a 1D array where the element is the layer
    #                   and the value is the number of neurons in that layer.
    #                   It is assumed that the matrix is built such that the 
    #                   first element is the input layer which is connected 
    #                   to the second to N elements for the hidden layer, 
    #                   and the last layer is the output layer.
    # mnfqF          == a 2D set of file names that points to an export as defined in the
    #                   export_network method for this network to import
    # mnfqT          == a 2D set of arrays of text that contains the export data of
    #                   a network, as defined in the export_network method
    # And the following optional arguments:
    # onehot         == change the inputs to onehot encoding. Index based by default
    # table          == a 2D array of the transition probability table to
    #                   initialize this for all 52 weeks of probabilities. 
    #                   The height/width must be equal to len(states) / len(actions) if passed
    # outputs        == an array of final populations for the states to let the
    #                   MNFQ networks compare to
    # seed           == a seed for the random object to produce results that can
    #                   be replicated
    # impossible     == if the table is not defined, this states which (if any) of the
    #                   transitions are NOT allowed
    **/
    @SuppressWarnings("removal")
	public mdq_learning(String[] states, String[] actions, Double learningR, Double learningRM, 
                 List<layer> networkL, int[] networkM, String networkF, String[] networkT,
                 List<layer> mnfqL, int[] mnfqM, String mnfqF, String[][][] mnfqT,
                 boolean onehot, double[][] table, double[] outputs, Integer seed, int[][] impossible) throws InvalidNeuronCount, NoLearningRate, InvalidLayerCount, IOException, InvalidLength, MissingArgument, NotEqualLength
    {
        
        //Check that the states and actions were passed
        if(states == null)
            throw new MissingArgument("The states have not been set");
        if(actions == null)
            throw new MissingArgument("The actions have not been set");

        //Check that the states and actions are valid
        if(states.length == 0)
            throw new InvalidLength("The amount of states must be greater than zero");
        if(actions.length == 0)
            throw new InvalidLength("The amount of states must be greater than zero");

    	//The number of states and actions should be equal
    	//The action performed is the new state
        if(states.length != actions.length) //NOTATION: I
    	    throw new InvalidLength("The number of states must equal the number of actions");
        
        //Check if the layers were passed
        if(networkL != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a layer array");

            this.network = new neural(learningR, null, null, networkL, null, null, null, seed);
        }
        //Else, check if we need to generate the network from the matrix
        else if(networkM != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a matrix");
            this.network = new neural(learningR, null, null, null, networkM, null, null, seed);
        }
        //Else, import the file
        else if(networkF != null)
            this.network = new neural(learningR, null, null, null, null, networkF, null, seed);
        
        //Else, import from array
        else
            this.network = new neural(learningR, null, null, null, null, null, networkT, seed);

        //Store the passed values
        this.states = states;
        this.actions = actions;
        this.onehot = onehot;

        //Set the seed if needed
        if(seed != null)
            this.rnd = new Random(seed);
        else
            this.rnd = new Random();

        //Check if the table was passed
        this.tables = new ArrayList<double[][]>();
        this.impossible = impossible;
        if(table != null)
        {
            //Make sure it is the correct size
            if(table.length != states.length || table[0].length != actions.length)
                throw new NotEqualLength("The passed table must be equal to the number of states by the number of actions");

            for (int i = 0; i < 52; i++)
            {
            	double[][] clone = new double[table.length][table[0].length];
            	for (int j = 0; j < table.length; j++)
            	{
            		for (int k = 0; k < table[0].length; k++)
            		{
            			boolean allowed = true;
                        if(impossible != null)
                        {
                        	for(int[] x : impossible)
                        	{
                        		if(x[0] == j && x[1] == k)
                        		{
                        			allowed = false;
                        			break;
                        		}
                        	}
                        }
            			
                        if (allowed)
                        {
                        	clone[j][k] = table[j][k];
                        }
                        else
                        {
                        	clone[j][k] = 0;
                        }
            		}
            	}
            	
                this.tables.add(clone);
            }
            this.random_table = false;
        }
        else
        {
            //Randomly initialize the table
        	table = new double[states.length][actions.length];
        	double[] temp = new double[states.length];

            for (int i = 0; i < states.length; i++)
            {
                for (int j = 0; j < actions.length; j++)
                {
                    //See if this is allowed
                	double val = 0.0;
                    if(impossible != null)
                    {
                    	boolean allowed = true;
                    	for(int[] x : impossible)
                    	{
                    		if(x[0] == i && x[1] == j)
                    		{
                    			allowed = false;
                    			break;
                    		}
                    	}
                    	if (!allowed) val = 0;
                    	else val = this.rnd.nextDouble();
                    }
                    else val = this.rnd.nextDouble();

                    table[i][j] = val;
                    temp[i] += table[i][j];
                }
            }
            
            //Normalize it
            for (int i = 0; i < states.length; i++)
            {
                double t = temp[i];
                for (int j = 0; j < actions.length; j++)
                    table[i][j] /= t;
            }

            //Add to the algorithm
            for (int i = 0; i < 52; i++)
            {
            	double[][] clone = new double[table.length][table[0].length];
            	for (int j = 0; j < table.length; j++)
            	{
            		for (int k = 0; k < table[0].length; k++)
            		{
            			clone[j][k] = table[j][k];
            		}
            	}
                this.tables.add(clone);
            }
            this.random_table = true;
        }

        //Initialize the mnfq network
        this.mnfq = new mnfq_iteration(new Integer(states.length), new Integer(actions.length), learningRM, mnfqL, mnfqM, mnfqF, mnfqT, outputs);
        this.mnfq.min_train_perc = this.min_train_perc;
    }
    
    /**
    # This method performs an epoch in the modified deep q-learning algorithm.
    # The population defines the population where each value corresponds to a member's state. 
    # This method returns the loss from the previous iteration + an array of state populations.
    **/
    @SuppressWarnings("removal")
	public Object[] perform_epoch(double[] population, int currentSubE, int epochsLeft, boolean learn, double[][] transitionChanges) throws OutputNotSet, NoConnectionException, InvalidLength, InvalidNeuronCount, NotEqualLength
    {
        //Check that the population is set
        if (population.length == 0)
            throw new InvalidLength("The population size must be positive");
        
        //Stores the current state of the table
        double[][] oldTable = new double[actions.length][states.length];
        
        //Store the q-values as well as rewards in a 3D array
        //The first dimension is the state, the second is the action- which stores the q-value or reward summation
        double[][] populationQs = new double[actions.length][states.length];
        
        double[][] populationRs = new double[actions.length][states.length];
        double[][] populationTrans = new double[actions.length][states.length];
        
        //Populate the old table and store it in the MNFQ algorithm
        int ind = (currentSubE % 52);
        double[][] table = this.tables.get(ind);
        
    	oldTable = new double[table.length][table[0].length]; //NOTATION: Q_t
    	for (int j = 0; j < table.length; j++)
    	{
    		for (int k = 0; k < table[0].length; k++)
    		{
    			oldTable[j][k] = table[j][k]; //NOTATION Q_{i, j, t} where j == i and k == j
    		}
    	}
    	
        //Get the current populations
        int[] statePopulations = new int[states.length]; //NOTATION: \hat{n}_{i, t}
        for (int i = 0; i < population.length; i++)
        {
            for (int j = 0; j < states.length; j++)
            {
                if(population[i] == j)
                {
                    statePopulations[j] += 1;
                    break;
                }
            }
        }
        
        //Run (population) number of sub-epochs
        List<Double> qVals = new ArrayList<Double>();

        //If the probability table was read from file or we're testing, use it perfectly
        //if((!this.network.random_restart_needed) && (!this.random_table || !learn))

        if (!this.random_table)
        {
	        int[][] partitions = new int[actions.length][states.length];
	        for (int i = 0; i < states.length; i++)
	        {
	            for (int j = 0; j < actions.length; j++)
	            {
	                //Determine number of individuals to move
	                partitions[i][j] = (int) Math.round(statePopulations[i] * table[i][j]);
	            }
	
	            //Ensure this adds up to the original pop
	            int diff = IntStream.of(partitions[i]).sum() - statePopulations[i];
	            if (diff != 0)
	            {
	                //Add to highest prob if not
	                double high_prob = -1.0;
	                int high_ind = -1;
	                for (int k = 0; k < actions.length; k++)
	                {
	                    if (table[i][k] > high_prob)
	                    {
	                        high_prob = table[i][k];
	                        high_ind = k;
	                    }
	                }
	                partitions[i][high_ind] += Math.abs(diff);
	            }
	        }
	        //Cycle each individual through the algorithm
	        for (int i = 0; i < states.length; i++) //NOTATION: s_i
	        {
	            for (int j  = 0; j < actions.length; j++)
	            {
	            	//Keep track of the MNFQ errors to do back-propagation afterwards
	            	Map<Double, List<Double>> mnfqHist = new HashMap<>();
	            	
	                for (int _p = 0; _p < partitions[i][j]; _p++)
	                {
	                	List<Double> info = new ArrayList<Double>();
	                	
	                    //Set the input of the network
	                    this.set_input(i, j, ind);
	                    
	                    info.add(new Double(i));
	                    info.add(new Double(j));
	                    
	                    //Propagate forward in the network
	                    this.network.propagate_forward(false);
	                    
	                    if(learn)
	                    {
	                        //Get the q-value and reward
	                        double qValue = this.network.get_network_output()[0];
	                        qValue = (Double.isNaN(qValue)) ? 0.0 : qValue;
	                        //Normalize it
	                        qValue = (qValue) / (qValue + table[i][j]);
	                        qValue = (Double.isNaN(qValue)) ? 0.0 : qValue;
	
	                        
	                        info.add(new Double(qValue));
	                        info.add(new Double(statePopulations[j]));
	
	                        Double error = new Double(-1);
	                        double[] temp = this.mnfq.calc_immediate_offset(i, j, oldTable, statePopulations[j], epochsLeft, qValue, false);
	
	                        error = new Double(temp[1]);
	                        
	                        //Store these in their array lists
	                        qVals.add(qValue);
	                        populationQs[i][j] += qValue;
	                        
	                        mnfqHist.put(new Double(Math.round(Math.abs(error)) + "." + i), info);
	                    }
	
	                    //Change the populations
	                    //Ensure this wouldn't put it below zero (rounding issue likely, happens with really low populations)
	                    if (statePopulations[i] != 0)
	                    {
	    	                statePopulations[i] -= 1;
	    	                statePopulations[j] += 1;
	                    }
	                }
	                
	                if (learn)
	                {
	    	            SortedSet<Double> sortedMnfq = new TreeSet<>(mnfqHist.keySet());
	    	            for (Double d : sortedMnfq)
	    	            {
	    	            	List<Double> curInfo = mnfqHist.get(d);
	    	            	
	    		            Double reward = new Double(-1);
	    		            double[] temp = this.mnfq.calc_immediate_offset((int)curInfo.get(0).doubleValue(), (int)curInfo.get(1).doubleValue(), oldTable, (int)curInfo.get(3).doubleValue(), epochsLeft, curInfo.get(2).doubleValue(), true);
	    		        
	    		            reward = new Double(temp[0]);
	    		            
	    		            if(reward.isNaN())
	    		                reward = new Double(0);
	    		            else if (reward.isInfinite())
	    		            {
	    		            	if(reward < 0)
	    		            		reward = new Double(-1);
	    		            	else
	    		            		reward = new Double(1);
	    		            }
	    		            populationRs[(int)curInfo.get(0).doubleValue()][(int)curInfo.get(1).doubleValue()] += reward;
	    		            populationTrans[(int)curInfo.get(0).doubleValue()][(int)curInfo.get(1).doubleValue()] += 1;
	    	            }
	                }
	            }
	        }
        }
        //Else, follow the random approach
        else
        {
        	//Keep track of the MNFQ errors to do back-propagation afterwards
        	Map<Double, List<Double>> mnfqHist = new HashMap<>();
        	
            for (int i = 0; i < population.length; i++) //NOTATION: a_{p, t}
            {
            	List<Double> info = new ArrayList<Double>();
            	 
                //Select a random action to take
                int state = (int)population[i]; //NOTATION: s_{p, t}
                int action = this.get_rand_action(state, table);
                
                info.add(new Double(state));
                info.add(new Double(action));
                
                //Set the input of the network
                this.set_input(state, action, ind);
                
                //Propagate forward in the network
                this.network.propagate_forward(false);
                
                if(learn)
                {
                    //Get the q-value and reward
                    double qValue = this.network.get_network_output()[0]; //NOTATION: q_{p, t}
                    qValue = (Double.isNaN(qValue)) ? 0.0 : qValue;
                    //Normalize it
                    qValue = (qValue) / (qValue + table[state][action]);
                    qValue = (Double.isNaN(qValue)) ? 0.0 : qValue;
                    
                    info.add(new Double(qValue));
                    info.add(new Double(statePopulations[action]));

                    Double error = new Double(-1);
                    double[] temp = this.mnfq.calc_immediate_offset(state, action, oldTable, statePopulations[action], epochsLeft, qValue, false);

                    error = new Double(temp[1]);
                    
                    //Store these in their array lists
                    qVals.add(qValue);
                    populationQs[state][action] += qValue;
                    
                    mnfqHist.put(new Double(Math.round(Math.abs(error)) + "." + i), info);
                }
                
                //Update the population member's state
                population[i] = action;

                //Change the populations
                //Ensure this wouldn't put it below zero (rounding issue likely, happens with really low populations)
                if (statePopulations[state] != 0)
                {
	                statePopulations[state] -= 1; //NOTATION: c_{i, t}
	                statePopulations[action] += 1;
                }
            }
         
            if (learn)
            {
	            SortedSet<Double> sortedMnfq = new TreeSet<>(mnfqHist.keySet());
	            for (Double d : sortedMnfq)
	            {
	            	List<Double> curInfo = mnfqHist.get(d);
	            	
		            Double reward = new Double(-1);
		            double[] temp = this.mnfq.calc_immediate_offset((int)curInfo.get(0).doubleValue(), (int)curInfo.get(1).doubleValue(), oldTable, (int)curInfo.get(3).doubleValue(), epochsLeft, curInfo.get(2).doubleValue(), true);
		        
		            reward = new Double(temp[0]);
		            
		            if(reward.isNaN())
		                reward = new Double(0);
		            else if (reward.isInfinite())
		            {
		            	if(reward < 0)
		            		reward = new Double(-1);
		            	else
		            		reward = new Double(1);
		            }
		            populationRs[(int)curInfo.get(0).doubleValue()][(int)curInfo.get(1).doubleValue()] += reward;
		            populationTrans[(int)curInfo.get(0).doubleValue()][(int)curInfo.get(1).doubleValue()] += 1;
	            }
            }
        }

        double final_res = 0.0;

        if(learn)
        {
            double[] sumS = new double[states.length];
            //Calculate the new q-values
            //Also, get sum for normalizing
            for (int i = 0; i < states.length; i++)
            {
                for (int j = 0; j < actions.length; j++)
                {
                    int multiplier = 1; // NOTATION \{eta}
                    Double offset = new Double(this.mnfq.calc_offset(i, j, oldTable, qVals.stream().mapToDouble(d -> d.doubleValue()).toArray(), statePopulations[j], epochsLeft));
                    // offset = 0.0;                    
                    if(offset.isNaN())
                        offset = new Double(0);
                    else if  (offset.isInfinite())
                    {
                        if(offset < 0)
                            offset = new Double(-1);
                        else
                            offset = new Double(1);
                    }

                    //Only change if this transition is possible
                    // NOTATION Equation 7
                    
                	boolean allowed = true;
                	if (impossible != null)
                	{
	                	for(int[] x : impossible)
	                	{
	                		if(x[0] == i && x[1] == j)
	                		{
	                			allowed = false;
	                			break;
	                		}
	                	}
                	}
                    
                    if(allowed)
                    {
                        table[i][j] += offset * this.network.get_learning_rate();
                        if(offset < 0) multiplier = -1;

                        if (populationTrans[i][j] != 0)
                        	table[i][j] += multiplier * ((populationRs[i][j] / populationTrans[i][j])) * this.network.get_learning_rate();
                    }
    
                    //Base case checking
                    //Seems to happen if initial population was zero
                    //Or offset is making the value negative
                    if(new Double(table[i][j]).isNaN())
                        table[i][j] = 0;

                    else if (new Double(table[i][j]).isInfinite())
                        table[i][j] = 0;

                    else if (table[i][j] < 0)
                        table[i][j] = -table[i][j];
                    
                    sumS[i] += table[i][j];
                    
                    transitionChanges[i][j] += oldTable[i][j] - table[i][j];
                }
            }

            //Normalize the q-values between zero and one
            for (int i = 0; i < states.length; i++)
            {
                double summ = sumS[i];
                for (int j = 0; j < actions.length; j++)
                    table[i][j] = table[i][j] / summ;
            }
            
            //Calculate the error
            double error = this.out_error(populationQs, population.length, table);
            
            //Set the error on the output neuron
            double[] err = {error};

            // BartG
            double [] pop_org = this.mnfq_outputs;
            double [] pop_new = Arrays.stream((int[]) statePopulations).asDoubleStream().toArray();

            for (int pi = 0; pi < pop_new.length; pi++){
                pop_new[pi] = (Double.isNaN(pop_new[pi])) ? 0.0 : pop_new[pi];
                pop_org[pi] = (Double.isNaN(pop_org[pi])) ? 0.0 : pop_org[pi];
            }


            if (weighted_err){
                final_res = this.weighted_diff(pop_org,pop_new);
            } else {
                final_res = this.mean_diff(pop_org,pop_new);
            }
            
            double [] pop_err = {final_res};

            if (this.use_error_metric.equals("accuracy")){
                this.network._set_output_error(pop_err);
            } else {
                this.network._set_output_error(err);
            }
            
            //Perform back-propagation
            this.network.propagate_backward(true);
        }
        
        this.tables.set(ind, table);
        if(learn)
        {
            //Get the loss
            this.loss_hold = this.avg_loss(table, oldTable); //NOTATION: E_t
            
            // BartG: always returning loss regardless of metric being used.
            Object[] ret = {this.loss_hold, statePopulations, transitionChanges};
            return ret;
        }
        else
        {
        	Object[] ret = {-1, statePopulations, transitionChanges};
            return ret;
        }
    }

    /**
    # This method sets the input of the network. The int values
    # correspond to an index in the states and actions arrays
    **/
    public void set_input(int state, int action, int week) throws InvalidNeuronCount
    {
        //Set the input
        double[] input = null;
        
        //Check input specification
        if(this.onehot)
        {
            input = new double[states.length + actions.length + 1];

            for (int i = 0; i < states.length; i++)
                input[i] = i == state ? 1 : 0;
            
            for (int i = 0; i < actions.length; i++)
                input[states.length + i] = i == action ? 1 : 0;
            
            input[input.length - 1] = week;
        }
        else
        {
            input = new double[3];
            
            input[0] = state;
            input[1] = action;
            input[2] = week;
        }
        
        this.network.set_input(input);
    }


    public static double sum(double...values) {
        double result = 0;
        for (double value:values)
            if (!Double.isNaN(value)){
                result += value;
            }
        return result;
     }
     public static double[]  weighted_sum(double...values) {
        double sum = sum(values);
        double [] results = new double[values.length];
        for (int i=0; i < values.length; i++)
            results[i] = values[i] / sum;
        return results;
    }


    public double  mean_diff(double [] values1, double [] values2) throws NotEqualLength{
        if (values1.length != values2.length){
            throw new NotEqualLength("The passed lists must be equal in length");
        }

        double [] results = new double[values1.length];
        for (int i=0; i < values1.length; i++){
            results[i] = Math.abs((values1[i]) - (values2[i])) / (values1[i]) - (values2[i]);
            if (this.use_error_adjust.equals("crossent")){
                results[i] = cross_entropy(1.0-results[i], results[i]);
            }

        }
        
        return sum(results);
    }
     public double  weighted_diff(double [] values1, double [] values2) throws NotEqualLength{
        if (values1.length != values2.length){
            throw new NotEqualLength("The passed lists must be equal in length");
        }
        double sum1 = sum(values1);
        double sum2 = sum(values2);
        double [] w1 = new double [values1.length];
        double [] w2 = new double [values2.length];
        for (int i=0; i < values1.length; i++){
            w1[i] = values1[i] / sum1;
            w2[i] = values2[i] / sum2;
        }

        double [] results = new double[values1.length];
        for (int i=0; i < values1.length; i++){
            //results[i] = Math.abs((w1[i]*values1[i]) - (w2[i]*values2[i])) / Math.max(sum1,sum2);
            
            results[i] = Math.abs(values1[i] - values2[i]) / Math.max(values1[i], values2[i]);
            results[i] = (Double.isNaN(results[i])) ? 0.0 : results[i];
            
            if (this.use_error_adjust.equals("crossent")){
                results[i] = cross_entropy(1.0-results[i], results[i]);
            }
        }
        
        //return sum(results);
        return sum(results) / results.length;
    }


    /**
    # This method uses roulette wheel to randomly select an action
    **/
    public int get_rand_action(int state, double[][] table)
    {
        //Pick a random number
        int sel = -1;
        double rand = this.rnd.nextDouble();
        
        //Determine which action to pick
        double temp = 0;
        double[] actions = table[state];
        for (int i = 0; i < actions.length; i++)
        {
            temp += actions[i];
            if(rand < temp)
            {
                sel = i;
                break;
            }
        }
        
        //If not valid, pick highest probability that is valid 
        int[] pair = {state, sel};
        if (this.impossible != null)
        {
        	boolean allowed = true;
        	for(int[] x : impossible)
        	{
        		if(x[0] == pair[0] && x[1] == pair[1])
        		{
        			allowed = false;
        			break;
        		}
        	}
        	
            if (!allowed)
            {
                int high_ind = -1;
                double high_prob = -1;

                for (int action = 0; action < actions.length; action++)
                {
                    pair[0] = state; pair[1] = action;
                    
                    allowed = true;
                	for(int[] x : impossible)
                	{
                		if(x[0] == pair[0] && x[1] == pair[1])
                		{
                			allowed = false;
                			break;
                		}
                	}
                    
                    if(allowed && actions[action] > high_prob)
                    {
                        high_prob = actions[action];
                        high_ind = action;
                    }
                }

                sel = high_ind;
            }
        }

        else if (table[state][sel] == 0)
		{
            int high_ind = -1;
            double high_prob = -1;

            for (int action = 0; action < actions.length; action++)
            {
                pair[0] = state; pair[1] = action;
                
                if(table[state][sel] != 0 && actions[action] > high_prob)
                {
                    high_prob = actions[action];
                    high_ind = action;
                }
            }

            sel = high_ind;
		}

        return sel;
    }


    /**
    # This method determines the error for the output neruon
    **/
    // BartG this is the loss
    // Calculating the loss for MDQL    
    public double out_error(double[][] popQs, int popSize, double[][] table)
    {
        double ret = 0;
        double tmp_ret = 0.0;
        
        //Run through each q-value and minus the average
        //population's q-value from it. Add that to a sum
        for (int i = 0; i < this.states.length; i++)
        {
            for (int j = 0; j < actions.length; j++){
                table[i][j] = (Double.isNaN(table[i][j])) ? 0.0 : table[i][j];
                popQs[i][j] = (Double.isNaN(popQs[i][j])) ? 0.0 : popQs[i][j];
                tmp_ret = (table[i][j] - (popQs[i][j] / popSize));
            
                if (this.use_error_adjust.equals("crossent")){
                    tmp_ret = cross_entropy(1.0-tmp_ret, tmp_ret);
                }
                ret += tmp_ret;
            }
        }
        
        //Divide the sum by the number of q-values for the average error
        ret /= (this.states.length * this.actions.length);
        
        return ret;
    }

    /**
    # This method determines the average loss for the population
    # based on the old q-values
    **/
    public double avg_loss(double[][] table, double[][] oldTable)
    {
    	double ret = 0;
    	
    	//Calculate the difference for each state
    	for (int i = 0; i < states.length; i++)
    	{
    		//Cycle through each action in the state
    		for (int j = 0; j < actions.length; j++)
    		{
    			//Add the difference onto the return value
                table[i][j] = (Double.isNaN(table[i][j])) ? 0.0 : table[i][j];
                oldTable[i][j] = (Double.isNaN(oldTable[i][j])) ? 0.0 : oldTable[i][j];
                ret += Math.abs(table[i][j] - oldTable[i][j]);
    		}
    	}
    	
    	//Divide it to get the average
    	ret /= (actions.length * states.length);
    	
    	return ret;
    }

    /**
    # This method sets the output of the MNFQ network. The int values
    # correspond to an index in the states and actions arrays
    **/
    public void set_output(double[] output)
    {
        this.mnfq_outputs = new double[output.length];
        for (int i=0; i<output.length; i++)
            this.mnfq_outputs[i] = output[i];

    	this.mnfq._set_output(output);
    }

    /**
    # This method sets the learning rate of the algorithm
    **/
    public void set_learning_rate(double lr,double lrm)
    {
        this.network.set_learning_rate(lr);
        this.mnfq._set_learning_rate(lrm);
    }
    
    public double[] get_learning_rate()
    {
    	double[] lrs = new double[2];
    	
    	lrs[0] = this.network.get_learning_rate();
    	lrs[1] = this.mnfq._get_learning_rate();
    	
    	return lrs;
    }
    
    public double cross_entropy(double yhat, double y){
        if (y == 1.0){
            return -1.0 * Math.log(yhat);
        } else {
            return -1.0 * Math.log(1.0 - yhat);
        }
    }

    public void adjust_learning_rate_decay(int epoch, double decay_l, double decay_m)
    {
    	this.network.adjust_learning_rate_decay(epoch, decay_l);
    	this.mnfq._adjust_learning_rate_decay(epoch, decay_m);
    }
    
    public void adjust_learning_rate_sched(int epoch, double decay_l, double decay_m, int patience_l, int patience_m)
    {
    	this.network.adjust_learning_rate_sched(epoch, decay_l, patience_l);
    	this.mnfq._adjust_learning_rate_sched(epoch, decay_m, patience_m);
    }

    @SuppressWarnings("removal")
	public Object[] adjust_random_restart(int epoch, List<Double> errors, int patience)
    {
	   boolean restarted = this.network.adjust_random_restart(epoch, errors, patience, false);
	   int tmp = this.mnfq._adjust_random_restart(epoch, patience, false);
	   
	   if (tmp > 0)
		   this.network.reset_learning_rate(epoch);
	  	   
	   this.mnfq._reset_errors();
	   
	   //Return result
	   Object[] ret = new Object[2];
	   
	   ret[0] = new Boolean(restarted);
	   ret[1] = new Integer(tmp);
	   
	   return ret;
    }
}
