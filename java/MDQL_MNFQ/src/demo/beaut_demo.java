package demo;
/*
# This file demonstrates the homelessness
# simulation referred to as MDQL with MNFQ
#
# Author: Andrew Fisher
*/

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import algorithm.adaptable_param;
import algorithm.beaut;
import algorithm.mdq_learning;
import exception.InvalidLength;
import exception.InvalidNeuronCount;
import exception.NoConnectionException;
import exception.NotEqualLength;
import exception.OutputNotSet;

public class beaut_demo
{
	@SuppressWarnings({ "removal", "unused" })
	public static void main(String[] args) throws Exception
	{
	        /*#############################################################################*/
	        //Declare whether or not to use pre-defined probabilities
	        boolean random_probs = false;
	
	        //Define learning rates and epochs to try out
	        double[] learningRates = new double[] {0.01};
	        int[] epochList = new int[] {1};
	
	        //Define a random seed to replicate results (set to None for a random seed)
	        Integer seed = 22;
	
	        //Declare whether or not onehot encoding should be used
	        boolean onehot = true;
	        
	        double learningRateM = 0.01;
	        
	        String dynamic_lr = "none";
	        /*#############################################################################*/
	
	        //Define the states and groups
	        String[] stateNames = new String[]{"S0", "S1", "S2", "S3", "S4"};
	        String[] groupNames = new String[]{"grp_one", "grp_two", "grp_three", "grp_four", "grp_five", "grp_six"};
	        
	        //Declare the directory to import the populations from for training
	        String importDir = "../../synthetic_data/raw_training_data";
	        
	        //Declare an initial transition probability table to speed up training (optional, set to None if not being used)
	        String probsFile = null;
	        if(random_probs == false)
	        {
	            probsFile = "../../synthetic_data/initial_matrix/grp_one_matrix.csv";
	        }
	
	        //If probabilities aren't defined, state if any transitions aren't allowed (optional, set to None if not being used)
	        Boolean impossible = null;
	        
	        //Run the algorithm
	        int states = stateNames.length;
	        for (double lr : learningRates)
	        {
	            for (int epochs : epochList)
	            {
	                System.out.println("Running the algorithm on the sanity dataset with:\nLearning rate (Lr)= \"" + Double.toString(lr));  
	                System.out.println("\"Training epoch length (E)= \"" + Integer.toString(epochs));
	                System.out.println("\"" + (probsFile == null ? "RANDOM" : "GIVEN") + " probabilities (P)"); 
	                System.out.println((seed == null ? "RANDOM seed (S)" : "Seed (S)= " + seed));
	                System.out.println((onehot ? "USING " : "NOT USING ") + "onehot encoding!");
	
	                //If set, read the initial transition probabilities
	                double[][] startingProbs = null;
	                if(probsFile != null)
	                {
	                    startingProbs = new double[states][states];
	                    boolean flag = true;
	                    int count = 0;
	                    
	                    File file = new File(probsFile);
	                    Scanner input = new Scanner(file);

                        while (input.hasNextLine())
                        {
                        	String line = input.nextLine();
                            if(flag)
                            {
                                flag = false;
                            }
                            else
                            {
                                String[] temp = line.split(",");
                                
                                for (int i = 0; i < states; i++)
                                {	
                                    startingProbs[count][i] = Float.parseFloat(temp[i + 1]) * 0.01;
                                }
                                
                                count += 1;
                            }
                        }
                        input.close();
	                }
	                
	                //Read the input data to a database
	                List<Map<String, double[][]>> database = new ArrayList<Map<String, double[][]>>();
	
	                File[] importFiles = new File(importDir).listFiles();
	                Arrays.sort(importFiles);
	                
	                Map<String, List<File>> groupFiles = new HashMap<String, List<File>>();
			        for(File file : importFiles)
			        {
			        	String group = null;
			        	String filename = file.getName();
			        	for(String g : groupNames)
			        	{
			        		if (filename.contains(g))
			        		{
			        			group = g;
			        			break;
			        		}
			        	}
			        	
			        	if(group == null)
			        	{
			        		throw new Exception("Unable to determine the group of file '" + filename + "'");
			        	}
			        	else
			        	{
			        		if (!groupFiles.containsKey(group))
			        		{
			        			groupFiles.put(group, new ArrayList<File>());
			        		}
			        		
			        		groupFiles.get(group).add(file);
			        	}
			        	
			        }
	                
	
					int max_week = -1;
					int totalDataPoints = 0;
	            	for (int g = 0; g < groupNames.length; g++)
	            	{
	            		String key = groupNames[g];
	            		List<File> curFiles = groupFiles.get(key);
	            		
	            		database.add(new HashMap<String, double[][]>());
		                for (File files : curFiles)
		                {
		                    //Next, add to the database
		                    double[] population = new double[states];
		                    
		                    int total = 0;
		                    int i = 0;
		                    
		                    BufferedReader f = new BufferedReader(new FileReader(files));
		                    String line = f.readLine();
		                    while(line != null)
		                    {
                            	int tmpPop = Integer.parseInt(line);
	                    		population[i] += tmpPop;
								totalDataPoints += tmpPop;
		                    	
		                        total += tmpPop;
		                        i += 1;
		                        
		                        line = f.readLine();
		                    }
		                    f.close();
		
		                    double[][] pop = new double[2][];

		                    double[] tempPop = new double[total];
		                    int tmp = 0;
		                    for (i = 0; i < states; i++)
		                    {
		                        for (int j = 0; j < population[i]; j++)
		                        {
		                            tempPop[tmp] = i;
		                            tmp += 1;
		                        }
		                    }
		
		                    pop[0] = population;
		                    pop[1] = tempPop;
							
		                    String numb = null;
		                    
		                    try
		                    {						
		                    	numb = Integer.parseInt(files.getName().split("_")[0]) + "";
		                    }
		                    catch (Exception e)
		                    {
		                    	numb = Integer.parseInt(files.getName().split("-")[0]) + "";
		                    }
		                    
	                        if (Integer.parseInt(numb) > max_week) {
	                            max_week = Integer.parseInt(numb);
	                        }

		                    database.get(g).put(numb, pop);
		                }
	            	}
					System.out.println("Total training datapoints read: " + totalDataPoints);
	
	                //Create the network to use
	                List<Integer> networkMatrix = new ArrayList<Integer>();

	                //Define the input layer
	                networkMatrix.add(onehot ? (states * 2) + 1 : 3);

	                //Define the hidden layers
	                int[] layerSizes_mdql = new int[] {4, 6, 8, 4};
	                for (int layerSize : layerSizes_mdql){
	                    networkMatrix.add(layerSize);
	                }
	                //Define the output layer
	                networkMatrix.add(1);

	                //Define the MNFQ networks' layouts
	               List<Integer> mnfqM = new ArrayList<Integer>();

	                //Define the input layer
	                mnfqM.add(3);

	                //Define two hidden layers with 6 and 12 neurons each respectively
	                //This seems to produce the best outputs from testing but, since the weights
	                //are randomly initialized, performance will vary
	                int[] layerSizes_mnfq = new int[] {6, 9, 12, 6};
	                for (int layerSize : layerSizes_mnfq){
	                    mnfqM.add(layerSize);
	                }

	                //Define the output layer
	                mnfqM.add(1);
	                
	                //Define MDQL instances for the groups
	                
	                mdq_learning[] mdqNetworks = new mdq_learning[groupNames.length];
	                
	                //Initialize the adaptable parameters for each group
	                adaptable_param[] exits_rate = new adaptable_param[groupNames.length];
	                adaptable_param[] spawn_rate = new adaptable_param[groupNames.length];
	                
	                for(int i = 0; i < groupNames.length; i++)
	                {
						mdqNetworks[i] = new mdq_learning(  stateNames, stateNames, 
															 new Double(lr), 
															 new Double(learningRateM), 
															 null,
															 networkMatrix.stream().mapToInt(d -> d.intValue()).toArray(), 
															 null, null, null,
															 mnfqM.stream().mapToInt(d -> d.intValue()).toArray(),
															 null, null,
															 onehot, startingProbs,
															 null,
															 seed, null    );

	                	mdqNetworks[i].use_error_adjust = "none";
	                	mdqNetworks[i].use_error_metric = "loss";
	                	mdqNetworks[i].min_train_perc = 20;
	                	mdqNetworks[i].mnfq.min_train_perc = 20;
	                	mdqNetworks[i].weighted_err = false;
	                	
	                	exits_rate[i] = new adaptable_param(2, 1, "ex_sigmoid", 0.01, new Integer(seed), null); //NOTATION: AP_x
	                	spawn_rate[i] = new adaptable_param(2, 1, "ex_sigmoid", 0.01, new Integer(seed), null); //NOTATION: AP_z
	                }

	                //Initialize the algorithm
	                beaut algorithm = new beaut(groupNames, stateNames, mdqNetworks);
	                
	                //Train the algorithm
	                for (int epoch_i = 0; epoch_i < epochs; epoch_i++)
	                {
	                	long start = System.nanoTime();

	                    //Cycle through all weeks
	                    @SuppressWarnings("unused")
						double loss = 0.0;
	                    int count = 0;

	                    double overallLoss = 0.0;
	                    
	                    double exitLoss = 0.0;
	                    double spawnLoss = 0.0;
	                    for (int j = 0; j < max_week; j++) //NOTATION: j == t
	                    {
	                    	for (int g = 0; g < groupNames.length; g++)
	                    	{
		                        //Set the output
		                    	//NOTATION: Note that j + 1 is used in some of the assignments below to represent the end of the current time period
	                    		double[] tempOut = null;
	                            try {
	                                tempOut = new double[database.get(g).get(j + 1 + "")[0].length];
	                            }
	                            catch (Exception e) {
	                                tempOut = new double[states_from_pop(new double[0], states).length];
	                            }
	                    		
		                    	for (int k = 0; k < tempOut.length; k++)
		                    	{
	                                try {
	                                    tempOut[k] = database.get(g).get(j + 1 + "")[0][k];
	                                }
	                                catch (Exception e) {
	                                    tempOut[k] = states_from_pop(new double[0], states)[k];
	                                }
		                    	}
		                        algorithm.set_output(g, tempOut);
		
		                        try {
	                                tempOut = new double[database.get(g).get(j + "")[1].length];
	                            }
	                            catch (Exception e) {
	                                tempOut = new double[pop_from_state(states_from_pop(new double[0], states), states).length];
	                            }
								
		                        //Run the sub-epochs
		                    	for (int k = 0; k < tempOut.length; k++)
		                    	{
	                                try {
	                                    tempOut[k] = database.get(g).get(j + "")[1][k];
	                                }
	                                catch (Exception e) {
	                                    tempOut[k] = pop_from_state(states_from_pop(new double[0], states), states)[k];
	                                }
		                    	}
		                    	
	                    		if (tempOut.length == 0) //Likely an excluded group
	                    		{
	                    			continue;
	                    		}
		                    	
		                        Object[] temp = algorithm.perform_epoch(g, tempOut, j, 1, true);
		                        // main_loss = algorithm.loss_hold;
		                        double lossT = (double) temp[0];
		                        int[] _statePopulations = (int[]) temp[1];
		                        
		                        loss += lossT;
		                        overallLoss += lossT;
		                        count += 1;
		
		                        if (dynamic_lr.equals("decay")){
		                            algorithm.adjust_learning_rate_decay(g, epoch_i, 0.5, 0.5);
		                        } else if (dynamic_lr.equals("schedule")){
		                            algorithm.adjust_learning_rate_sched(g, epoch_i, 0.5, 0.5, 5, 5);
		                        }
		
		                        //Train the adaptable parameters
								double actual_pop = -1.0;
								try {
									actual_pop = DoubleStream.of(database.get(g).get(j + 1 + "")[0]).sum();
								}
								catch (Exception e) {
									actual_pop = DoubleStream.of(states_from_pop(new double[0], states)).sum();
								}
								
								double sim_pop = IntStream.of(_statePopulations).sum(); //NOTATION: \hat{N}_t
		
		                        //Determine the percent needed/removed
		                        double exits_percent = (sim_pop - actual_pop) / sim_pop; //NOTATION: x_t
		                        double spawn_percent = (actual_pop - sim_pop) / actual_pop; //NOTATION: z_t
		
		                        //Set the parameter's outputs
		                        double[] tempExitO = {exits_percent};
		                        exits_rate[g].set_output(tempExitO);
		                        
		                        double[] tempSpawnO = {spawn_percent};
		                        spawn_rate[g].set_output(tempSpawnO);
		
		                        //Set their inputs
		                        int week = (j % 52);
		                        
		                        double[] tempExitI = {week, sim_pop};
		                        exits_rate[g].set_input(tempExitI);
		                        
		                        double[] tempSpawnI = {week, sim_pop};
		                        spawn_rate[g].set_input(tempSpawnI);
		
		                        //Train them
		                        Object[] tmp = exits_rate[g].process_input(true);
		                        double[] errTmp = (double[])tmp[1];
		                        
		                        exitLoss += errTmp[0];
		                        
		                        tmp = spawn_rate[g].process_input(true);
		                        errTmp = (double[])tmp[1];
		                        
		                        spawnLoss += errTmp[0];
	                    	}
	                    }
	                }
	                System.out.println("\nComplete!\n");
	                
	              //Test the algorithm
	                System.out.println("Testing the algorithm with the initial population:");

	                //Get the initial population
	                double[] tmp = new double[states];
	                
	                for (int g = 0; g < groupNames.length; g++)
	                {
	                	int length = -1;
	                	double[] pop = null;
	                	
	                	try
	                	{
	                		pop = database.get(g).get("0")[0];
	                	}
	                	catch (Exception e)
	                	{
	                		pop = states_from_pop(new double[0], states);
	                	}
	                	length = pop.length;
	                	
		                for (int i = 0; i < length; i++)
		                {
	                		tmp[i] += pop[i];
		                }
	                }
	                
	                for (int i = 0; i < tmp.length; i++)
	                {
	                	System.out.println(stateNames[i] + ": " + (tmp[i]));
	                }

	                //Run the sub-epochs
	                List<double[]> graph = new ArrayList<double[]>();
	                
	                double[] population = pop_from_state(tmp, states);
	                graph.add(states_from_pop(tmp, states));
	                for (int i = 0; i < database.get(0).size() - 1; i++)
	                {
	                	population = null;
	                	for (int g = 0; g < groupNames.length; g++)
	                	{
	                		if (population == null)
	                		{
	                            try {
	                            	population = database.get(g).get(i + "")[0];
	                            }
	                            catch (Exception e9) {
	                            	population = states_from_pop(new double[0], states);
	                            }
	                		}
	                		
		                	double[] tempOut = new double[population.length];
		                	for (int j = 0; j < tempOut.length; j++)
		                		tempOut[j] = population[j];
		                	Object[] temp = algorithm.perform_epoch(g, tempOut, i, 1, false);
		                    //double _ = (double) temp[0];
		                    population = Arrays.stream((int[]) temp[1]).asDoubleStream().toArray();
		                    for (int pi = 0; pi < population.length; pi++)
		                       population[pi] = (Double.isNaN(population[pi])) ? 0.0 : population[pi];
		
		                    //Apply the adaptable parameters to the population
		                    int week = (i % 52);
		                    population = apply_exits(population, week, exits_rate[g]);
		                    population = apply_spawn(population, week, spawn_rate[g]);
	                	}
	                	
	                    graph.add(algorithm.get_total_population());
	                }

	                //Display final results
	                System.out.println("\nFinal populations: ");
	                population = algorithm.get_total_population();
	                for (int i = 0; i < states; i++)
	                	System.out.println(stateNames[i] + ": " + (population[i]));
	                
	                System.out.println("Complete!\n");
	            }
	        }
	}
	
	
	/**
	# This method applies the exits parameter to the
	# population and returns the result
	**/
	public static double[] apply_exits(double[] population, int week, adaptable_param exits) throws InvalidNeuronCount, OutputNotSet, NoConnectionException, InvalidLength, NotEqualLength
	{
	    //Normalize the population
		double[] norm = new double[population.length];
		for (int ind = 0; ind < population.length; ind++)
		{
			int i = (int)population[ind];
			norm[ind] = i / DoubleStream.of(population).sum();
		}
	
	    //Determine how many individuals per state need to exit
		double[] temp = {week, DoubleStream.of(population).sum()};
	    exits.set_input(temp);
	    
	    double[] tmp = (double[])exits.process_input(false)[0];
	    
	    double exits_percent = tmp[0];
	    double total_exited = DoubleStream.of(population).sum() * exits_percent;
	    
	    if(total_exited < (DoubleStream.of(population).sum() / 2))
	    {
	        for (int i = 0; i < population.length; i++)
	        {
	            population[i] -= (int)Math.abs(norm[i] * total_exited);
	           //Ensure it didn't go below zero
	            if (population[i] < 0)
	            {
	            	population[i] = 0;
	            }
	        }
	    }
	
	    return population;
	}
	
	/**
	# This method applies the spawn parameter to the
	# population and returns the result
	**/
	public static double[] apply_spawn(double[] population, int week, adaptable_param spawn) throws InvalidNeuronCount, OutputNotSet, NoConnectionException, InvalidLength, NotEqualLength
	{
	    //Normalize the population
		double[] norm = new double[population.length];
		for (int ind = 0; ind < population.length; ind++)
		{
			int i = (int)population[ind];
			norm[ind] = i / DoubleStream.of(population).sum();
		}
	
	    //Determine how many individuals per state need to spawn
		double[] temp = {week, DoubleStream.of(population).sum()};
	    spawn.set_input(temp);
	    
	    double[] tmp = (double[])spawn.process_input(false)[0];
	    
	    double spawn_percent = tmp[0];
	    double total_spawned = DoubleStream.of(population).sum() * spawn_percent;
	
	    if(total_spawned < (DoubleStream.of(population).sum() / 2))
	    {
	        for (int i = 0; i < population.length; i++)
	            population[i] += (int)Math.abs(norm[i] * total_spawned);
	    }
	
	    return population;
	}
	
	/*
	# This method takes a population of individuals
	# and converts them to an array of state populations
	*/
	public static double[] states_from_pop(double[]population, int num_states)
	{
	    double[] ret_pop = new double[num_states];
	
	    for (int i = 0; i < num_states; i++)
	    {
	        int count = 0;
	        for (int j = 0; j < population.length; j++)
	            count += (population[j] == i  ? 1 : 0);
	        ret_pop[i] = count;
	    }
	
	    return ret_pop;
	}
	
	
	/*
	# This method takes a set of state populations
	# and converts them to an array of individuals
	*/
	public static double[] pop_from_state(double[] population, int num_states)
	{
	    int total = (int)DoubleStream.of(population).sum();
	    double[] tempPop = new double[total];
	    int tmp = 0;
	    
	    for (int i = 0; i < num_states; i++)
	    {
	        for (int _j = 0; _j < population[i]; _j++)
	        {
	            tempPop[tmp] = i;
	            tmp += 1;
	        }
	    }
	
	    return tempPop;
	}
}