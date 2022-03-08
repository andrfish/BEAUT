/**
# This file represents a neural network
#
# Author: Andrew Fisher
**/

package neural;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import exception.*;

public class neural
{	
	private Double learningRate = null;
	private Double init_learningRate = null;
	private int reset_epoch = 0;
	
	private List<layer> networkLayers = null;
	
	private int size = 0;
	public double[] output = null;
	
	public List<Double> previousErrors = new ArrayList<Double>();
	
    /**
    # This method initalizes the network with the following arguments:
    # learningRate   == the learning rate for the network. An optional
    #                   argument IFF you pass the networkF argument; if 
    #                   it is passed with networkF, it will override
    #                   whatever value it reads from the file
    # inputA         == the input of the network. An optional argument
    #                   but it must be set before propagating forward
    # outputA        == the output to compare the network's to. 
    #                   An optional argument but it must be set before
    #                   propgating forward
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
    # And the following optional argument:
    # seed           == a seed for the random generator
    **/
    @SuppressWarnings("removal")
	public neural(Double learningR, double[] inputA, double[] outputA, List<layer> networkL, int[] networkM, String networkF, String[] networkT, Integer seed) throws InvalidNeuronCount, NoLearningRate, InvalidLayerCount, IOException
    {
        //Check if the layers were passed
        if(networkL != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a layer array");
            
            this.learningRate = learningR;
            this.init_learningRate = new Double(learningR.doubleValue());
            this.networkLayers = networkL;
        }

        //Else, check if we need to generate the network from the matrix
        else if(networkM != null)
        {
            //Check that the learning rate was passed
            if(learningR == null)
                throw new NoLearningRate("The learning rate must be passed if creating the network from a matrix");
            
            this.learningRate = learningR;
            this.init_learningRate = new Double(learningR.doubleValue());
            this.networkLayers = new ArrayList<layer>();

            //Make sure that the number of layers is valid
            //This implementation assumes that you will have, at least, an
            //input layer and an output layer
            if (networkM.length < 2)
                throw new InvalidLayerCount("There must be at least two layers in the network");

            for (int i = 0; i < networkM.length; i++)
            {
                //Make sure that the number of neurons is valid
                if (networkM[i] <= 0)
                    throw new InvalidNeuronCount("The number of neurons (" + networkM[i] + ") in layer \"" + (i + 1) + "\"  must be greater than zero");

                //Create a new layer
                layer curLayer = new layer(networkM[i], this.learningRate, null, null, seed);
                
                //Set it as the next layer for the previous if applicable
                if(i != 0)
                    this.networkLayers.get(i - 1).set_next_layer(curLayer, true);
                
                //Add it to the array of layers
                this.networkLayers.add(curLayer);
            }
        }
        
        //Else, import the file
        else if (networkF != null)
        {
            this.import_network(networkF, null);
            if(learningR != null)
            {
                this.set_learning_rate(learningR);
                this.init_learningRate = new Double(learningR.doubleValue());
            }
        }
        
        //Else, import from array
        else
        {
            this.import_network(null, networkT);
            if(learningR != null)
            {
                this.set_learning_rate(learningR);
                this.init_learningRate = new Double(learningR.doubleValue());
            }
        }

        this.size = this.networkLayers.size();

        if(outputA != null)
            this.set_output(outputA);
        else
            this.output = null;
        
        if(inputA != null)
            this.set_input(inputA);
    }

    /**
    # This method propagates forward through the network and returns
    # the mean squared error for the output if errorRet is True
    **/
    public Double propagate_forward(boolean errorRet) throws NotEqualLength, NoConnectionException, OutputNotSet
    {
        //Check that the output has been set if returning the error
        if (this.output == null && errorRet)
            throw new OutputNotSet("The output for the network to compare to has not been set");

        //Propagate through the network
        for (int i = 0; i < this.size - 1; i++)
            this.networkLayers.get(i).forward();

        if (errorRet)
        {
            //Set the error for the output neurons
            layer outputLayer = this.networkLayers.get(this.size - 1);
            List<neuron> outputNeurons = outputLayer.get_neuron_list();
            double[] networkOutput = this.get_network_output();
            for (int i = 0; i < outputNeurons.size(); i++)
                outputNeurons.get(i).set_error(this.output[i] - networkOutput[i]);

            //Return the mean squared error
            double err = error.mse(this.output, networkOutput);
            previousErrors.add(err);
            return err;
        }
        return null;
    }

    /**
    # This method propagates backward through the network. Assumes to 
    # train the network by default
    **/
    public void propagate_backward(boolean learn) throws OutputNotSet, NoConnectionException
    {
        //Propagate through the network
        for (int i = this.size - 2; i > -1; i--)
            this.networkLayers.get(i).backward(learn);
    }

    /**
    # This method exports the network to an array and/or file depending
    # on the writeToFile argument. If it is true, the filename argument
    # must be passed.
    # First line:        number of layers
    # Second line:       the learning rate
    # Third line:        number of neurons in the input layer (refer to as nI)
    # Fourth to nI line: each neuron will have 5 lines. The first line will
    #                    be the number of weights. The second line will be
    #                    the weights, each separated by a space. The third
    #                    line will be the bias, the fourth the output, and
    #                    the fith the error.
    # The rest of the file will follow a similar format from the third
    # line to the fourth to nO line definitions above until the output
    # layer has been written. Note that the output is NOT exported
    **/
    public String[] export_network(boolean writeToFile, String filename) throws IOException, NoFilenameSpecified
    {
        List<String> ret = new ArrayList<String>();

        if(writeToFile)
        {
            // Check that the filename was passed
            if(filename == null)
                throw new NoFilenameSpecified("The filename was not specified to export to");
        }

        // Add the number of layers
        ret.add(size + "");

        // Add the learning rate
        ret.add(learningRate + "");

        // Add each layer
        for (int i = 0; i < size; i++)
        {
            layer curLayer = networkLayers.get(i);

            // Add number of neurons
            ret.add(curLayer.count + "");

            // Add each neuron in the layer
            for (int j = 0; j < curLayer.count; j++)
            {
                neuron curNeuron = curLayer.get_neuron_list().get(j);

                // Add the number of weights
                
                ret.add((curNeuron.weightList == null ? 0 : curNeuron.weightList.length) + "");

                // Add each weight
                if(curNeuron.weightList != null)
                {
                    String temp = "";
                    for (int k = 0; k < curNeuron.weightList.length; k++)
                        temp += curNeuron.weightList[k] + " ";
                    temp = temp.trim();

                    // Add weights
                    ret.add(temp);
                }
                // Add bias
                if(curNeuron.bias != Double.MIN_VALUE)
                    ret.add(curNeuron.bias + "");
                else
                    ret.add("None");
                
                // Add output
                if(curNeuron.output != Double.MIN_VALUE)
                    ret.add(curNeuron.output + "");
                else
                    ret.add("None");
                
                // Add error
                if(curNeuron.error != Double.MIN_VALUE)
                    ret.add(curNeuron.error + "");
                else
                    ret.add("None");
            }
        }
        
        if(writeToFile)
        {
            // Write to file
        	File file = new File(filename);
        	try 
        	{
        		file.getParentFile().mkdirs();
        	}
        	catch (Exception e){}
        	FileWriter writer = new FileWriter(file);
        	
        	BufferedWriter output = new BufferedWriter(writer);
            for(String line : ret)
            {
                // Write line to output file
                output.write(line);
                output.newLine();
            }
            output.flush();
            output.close();
        }

        return ret.toArray(new String[ret.size()]);
    }

    /**
    # This method imports a network from a text file based on the export
    # method above. It will overwrite everything in this neural network.
    # Pass ONE of the following arguments:
    # inputF    == the filename to read the input from
    # inputT    == the array of text that is of equivalent format as defined
    #              in the export_network method
    **/
    public void import_network(String inputF, String[] inputT) throws InvalidLayerCount, InvalidNeuronCount, IOException
    {
        List<layer> networkLayers = new ArrayList<layer>();
        output = null;

        // Read the size and learning rate first
        // Then, put the rest into an array to deal with easily
        List<String> inArray = new ArrayList<String>();
        if(inputF != null)
        {
            int count = 0;
            BufferedReader f = new BufferedReader(new FileReader(inputF));
            String line = f.readLine();
            while(line != null)
            {
                if (count == 2)
                {
                    // Read the rest of the file into an array
                    inArray.add(line);
                }
                else if (count == 0)
                {
                    // Read the number of layers
                    size = Integer.parseInt(line);
                    count += 1;
                }
                else if (count == 1)
                {
                    // Read the learning rate
                    learningRate = Double.parseDouble(line);
                    count += 1;
                }
                    
                line = f.readLine();
            }
            f.close();
        }
        else
        {
            inArray = Arrays.asList(inputT);
            
            //Read size and learning rate
            size = Integer.parseInt(inArray.get(0));
            learningRate = Double.parseDouble(inArray.get(1));
            
            //Remove those values from the array
            inArray.remove(0);
            inArray.remove(0);
        }

        // Create the network
        int count = 0;

        // Read each layer
        for (int i = 0; i < size; i++)
        {
            //Read the number of neurons in the layer
            int layerSize = Integer.parseInt(inArray.get(count));
            count += 1;

            // Make sure it's greater than 0
            if (layerSize < 1)
                throw new InvalidNeuronCount("The number of neurons must be greater than zero");

            // Read each neuron in the layer
            List<neuron> neurons = new ArrayList<neuron>();
            for (int j = 0; j < layerSize; j++)
            {
                int numWeights = Integer.parseInt(inArray.get(count));
                count += 1;

                // First, read the weights
                List<Double> weights = null;
                if(numWeights > 0)
                {
                	weights = new ArrayList<Double>();
                    String[] temp = inArray.get(count).split(" ");
                    for (int k = 0; k < temp.length; k++)
                        weights.add(Double.parseDouble(temp[k]));
                    count += 1;
                }
                
                // Next, read the bias
                double bias = Double.MIN_VALUE;
                if(!inArray.get(count).equals("None"))
                    bias = Double.parseDouble(inArray.get(count));
                count += 1;

                // Next, read the output
                double output = Double.MIN_VALUE;
                if(!inArray.get(count).equals("None"))
                    output = Double.parseDouble(inArray.get(count));
                count += 1;

                // Lastly, read the error
                double error = Double.MIN_VALUE;
                if(!inArray.get(count).equals("None"))
                    error = Double.parseDouble(inArray.get(count));
                count += 1;

                // Create the neuron
                neuron layerNeuron = new neuron((weights == null ? null : weights.stream().mapToDouble(d -> d).toArray()), bias);
                
                // Set the output
                if(output != Double.MIN_VALUE)
                    layerNeuron.output = output;
                
                // Set the error
                if(error != Double.MIN_VALUE)
                    layerNeuron.error = error;

                // Add to the neuron list
                neurons.add(layerNeuron);
            }

            // Create the layer
            layer newLayer = new layer(layerSize, learningRate, neurons, null, null);

            // Link to previous layer if applicable
            if (networkLayers.size() >= 1)
                networkLayers.get(networkLayers.size() - 1).set_next_layer(newLayer, false);
            
            // Add to layers
            networkLayers.add(newLayer);
        }
        this.networkLayers = networkLayers;
    }

    /**
    # This method sets the input for the network
    **/
    public void set_input(double[] input) throws InvalidNeuronCount
    {
        //Check that the number of inputs matches the number
        //of input neurons
        if(input.length != this.networkLayers.get(0).count)
            throw new InvalidNeuronCount("The number of neurons (" + this.networkLayers.get(0).count + ") does not match the number of inputs passed (" + input.length + ")");

        //Set the input
        List<neuron> inputNeurons = this.networkLayers.get(0).get_neuron_list();
        for (int i = 0; i < this.networkLayers.get(0).count; i++)
            inputNeurons.get(i).set_output(input[i]);
    }

    /**
    # This method sets the output to compare the network to
    **/
    public void set_output(double[] outputA) throws InvalidNeuronCount
    {
        //Check that the output equals the number of output neurons
        if(outputA.length != this.networkLayers.get(this.size - 1).count)
            throw new InvalidNeuronCount("The number of neurons (" + this.networkLayers.get(this.size - 1).count + ") does not match the number of outputs passed (" + outputA.length + ")");
        
        //Set the output
        this.output = outputA;
    }

    /**
    # This method sets the learning rate for all layers
    **/
    public void set_learning_rate(double learningR)
    {
        this.learningRate = learningR;

        for (int i = 0; i < this.networkLayers.size(); i++)
            this.networkLayers.get(i).learningRate = this.learningRate;
    }
    
    public double get_learning_rate()
    {
        return this.learningRate;
    }
    
    public void adjust_learning_rate_decay(int epoch, double decay){
    	epoch -= reset_epoch;
    	set_learning_rate(this.init_learningRate * (1.0 / (1.0 + decay * epoch)));
    }        
    
    public void adjust_learning_rate_sched(int epoch, double decay, int patience){
    	epoch -= reset_epoch;
        if ((epoch % patience) == 0){
        	set_learning_rate(this.init_learningRate * (1.0 / (1.0 + decay * epoch)));
        }
    }
    
    public boolean adjust_random_restart(int epoch, List<Double> errors, int patience, boolean lastLayerOnly){
        List<Double> last_errors = new ArrayList<Double>();
        boolean random_restart_needed = false;	
        
        if (((epoch+1) % patience) == 0 && errors.size() >= patience){
            for (int i = errors.size() - patience; i < errors.size(); i++){
                last_errors.add(errors.get(i));
            }

            double error_slope = getSlope(last_errors.stream().mapToDouble(d -> d).toArray());
            if (error_slope >= 0){
                random_restart_needed = true;
                reset_epoch = epoch;
                reset_weights(lastLayerOnly);
            }
        }
        
        return random_restart_needed;
    }
    
    public static double getSlope(double[] y_values) {
        double [] x_values = new double [y_values.length];
        for (int i = 0; i < (y_values.length - 1); i++)
            x_values[i] = i;

        double slope = 0;

        for (int i = 0; i < (x_values.length - 1); i++) {
            double y_2 = y_values[i + 1];
            double y_1 = y_values[i];

            double delta_y = y_2 - y_1;

            double x_2 = x_values[i + 1];
            double x_1 = x_values[i];

            double delta_x = x_2 - x_1;

            slope += delta_y / delta_x;
        }

        return slope / (x_values.length);
    }

    /**
    # This method gets the output for the network
    **/
    public double[] get_network_output()
    {
        layer outputLayer = this.networkLayers.get(this.size - 1);
        List<neuron> outputNeurons = outputLayer.get_neuron_list();
    	
        double[] ret = new double[outputNeurons.size()];
        		
        for (int i = 0; i < outputNeurons.size(); i++)
            ret[i] = outputNeurons.get(i).get_output();
            
        return ret;
    }

    /**
    # This method sets the error for the output layer
    **/
    public void _set_output_error(double[] errors) throws InvalidLength
    {
        //Set the error for the output neurons
        layer outputLayer = this.networkLayers.get(this.size - 1);
        List<neuron> outputNeurons = outputLayer.get_neuron_list();
        
        //Check that the length of errors is correct
        if(errors.length != outputNeurons.size())
        	throw new InvalidLength("The amount of errors must be equal to the amount of output neurons");
        
        for (int i = 0; i < outputNeurons.size(); i++)
            outputNeurons.get(i).error = errors[i];
    }

    /**
    # This method sets the output for the output layer
    **/
    public void _set_output(double[] outputs) throws InvalidLength
    {
        //Set the error for the output neurons
        layer outputLayer = this.networkLayers.get(this.size - 1);
        List<neuron> outputNeurons = outputLayer.get_neuron_list();
        
        //Check that the length of outputs is correct
        if(outputs.length != outputNeurons.size())
        	throw new InvalidLength("The amount of outputs must be equal to the amount of output neurons");
        
        for (int i = 0; i < outputNeurons.size(); i++)
            outputNeurons.get(i).set_output(outputs[i]);
    }
    
    public void reset_weights(boolean lastLayerOnly)
    {
    	if (lastLayerOnly)
    	{
    		this.networkLayers.get(this.networkLayers.size() - 2).randomize_weights();
    	}
    	else
    	{
        	for(layer l : this.networkLayers)
        	{
        		l.randomize_weights();
        	}
    	}
    	
    	reset_learning_rate(this.reset_epoch);
    }
    
    public void reset_learning_rate(int epoch)
    {
    	this.reset_epoch = epoch;
    	this.learningRate = this.init_learningRate;
    }
}