/**
# This file demonstrates the BEAUT model
#
# Author: Andrew Fisher
**/

package algorithm;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import exception.NoFilenameSpecified;

public class beaut {
	private String[] groupNames = null;
	private String[] states = null;
	public mdq_learning[] mdqNetworks = null;
	
	private int[][] groupPopulations = null;
	
	//This stores the total change in each transition probability for each week in the year
	//By total I mean for all epochs. Therefore, to get the average change
	//for each probability for each week, you would divide the final values
	//by the number of epochs it was trained for.
	public List<List<double[][]>> groupWeeklyMatrixChanges = null;
	
    /**
    # This method initializes the BEAUT model with the following arguments:
    # groupNames         == a list of the names for each agent-group
    # states			 == a list of the names for each state
    # mdqNetworks		 == a list of initialized MDQL networks for each agent-group
    # Note that each group will be referenced by their index in groupNames
    **/
    public beaut(String[] groupNames, String[] states, mdq_learning[] mdqNetworks)
    {
    	this.groupNames = groupNames.clone();
    	this.states = states;
    	this.mdqNetworks = mdqNetworks;
    	
    	// Initialize the populations
    	groupPopulations = new int[groupNames.length][states.length];
    	groupWeeklyMatrixChanges = new ArrayList<List<double[][]>>();
    	
    	for(int g = 0; g < groupNames.length; g++)
    	{
    		groupWeeklyMatrixChanges.add(new ArrayList<double[][]>());
    		
    		for (int w = 0; w < 52; w++)
    		{
    			groupWeeklyMatrixChanges.get(g).add(new double[states.length][states.length]);
    		}
    	}
    }
    
    public Object[] perform_epoch(int group, double[] population, int currentSubE, int epochsLeft, boolean learn) throws Exception
    {   	
    	//Perform an epoch
    	Object[] ret = this.mdqNetworks[group].perform_epoch(population, currentSubE, epochsLeft, learn, groupWeeklyMatrixChanges.get(group).get(currentSubE % 52));
    	
    	//Store the result
    	groupPopulations[group] = (int[])ret[1];
    	groupWeeklyMatrixChanges.get(group).set(currentSubE % 52, (double[][])ret[2]);
    	
    	//Return the result
    	return ret;
    }
    
    public void set_output(int group, double[] values)
    {
    	this.mdqNetworks[group].set_output(values);
    }
    
    public double[] get_population(int g)
    {
    	double[] ret = new double[this.states.length];
    	
    	for(int i = 0; i < this.groupPopulations[g].length; i++)
    	{
    		ret[i] = this.groupPopulations[g][i];
    	}
    	
    	return ret;
    }
    
    public double[] get_total_population()
    {
    	double[] ret = new double[this.states.length];
    	
    	for(int i = 0; i < this.groupNames.length; i++)
    	{
    		int[] pop = this.groupPopulations[i];
    		for(int j = 0; j < this.states.length; j++)
    		{
    			ret[j] += pop[j];
    		}
    	}
    	
    	return ret;
    }
    
    public void adjust_learning_rate_decay(int group, int epoch, double decay_l, double decay_m)
    {
    	this.mdqNetworks[group].adjust_learning_rate_decay(epoch, decay_l, decay_m);
    }
    
    public void adjust_learning_rate_sched(int group, int epoch, double decay_l, double decay_m, int patience_l, int patience_m)
    {
    	this.mdqNetworks[group].adjust_learning_rate_sched(epoch, decay_l, decay_m, patience_l, patience_m);
    }

    public void adjust_random_restart(int epoch, List<Double> errors, int patience)
    {
    	int mdql_restart_cnt = 0;
    	int mnfq_restart_cnt = 0;
    	
    	for(mdq_learning network : this.mdqNetworks)
    	{
    	   Object[] res = network.adjust_random_restart(epoch, errors, patience);
    	   
    	   boolean restarted = (Boolean)res[0];
    	   int tmp = (Integer)res[1];
    	   
    	   if (restarted)
    		   mdql_restart_cnt += 1;
    	   mnfq_restart_cnt += tmp;
    	}
    	
    	System.out.println("\nA total of " + mdql_restart_cnt + "/" + this.mdqNetworks.length + " MDQL networks were reset!");
    	System.out.println("A total of " + mnfq_restart_cnt + "/" + (this.mdqNetworks.length * this.states.length * this.states.length) + " MNFQ networks were reset!\n");
    }
    
    public void export_networks(boolean writeToFile, String filenamePrefix) throws IOException, NoFilenameSpecified
    {
    	for(int g = 0; g < this.mdqNetworks.length; g++)
    	{
    		this.mdqNetworks[g].network.export_network(writeToFile, filenamePrefix + "_" +  g + ".txt");
    	}
    }
    
    public void export_mnfq_networks(boolean writeToFile, String directoryPrefix) throws IOException, NoFilenameSpecified
    {
    	for(int g = 0; g < this.mdqNetworks.length; g++)
    	{
    		mdq_learning algorithm = this.mdqNetworks[g];
    		
            for (int i = 0; i < algorithm.mnfq.networks.length; i++)
            {
            	for (int j = 0; j < algorithm.mnfq.networks[i].length; j++)
            	{
            		algorithm.mnfq.networks[i][j].export_network(true, directoryPrefix + "mnfq_" + g + "/mnfq[" + i + "][" + j + "].txt");
            	}
            }
    	}
    }
    
    public double[] get_average_learning_rates()
    {
    	double[] ret = new double[2];
    	
    	double mdq_lr = 0.0;
    	double mnfq_lr = 0.0;
    	
    	for (mdq_learning network : this.mdqNetworks)
    	{
    		double[] tmp = network.get_learning_rate();
    		
    		mdq_lr += tmp[0];
    		mnfq_lr += tmp[1];
    	}
    	
    	//Average them
    	mdq_lr /= this.mdqNetworks.length;
    	mnfq_lr /= this.mdqNetworks.length;
    	
    	//Return the results
    	ret[0] = mdq_lr;
    	ret[1] = mnfq_lr;
    	
    	return ret;
    }
}
