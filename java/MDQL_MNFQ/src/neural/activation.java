/**
# This class is used for activation functions in my neural network
# code
#
# Author: Andrew Fisher
**/

package neural;

public class activation
{
	/**
	# This implements the sigmoid activation function
	**/
	public static double sigmoid(double x)
	{
		double y = 0;
	    try
	    {
	        y = 1 / (1 + Math.exp(-x));
	    }
	    catch (Exception e) //X is too large
	    {
	        if(x > 0)
	            y = 1;
	        else
	            y = 0;
	    }
	
	    return y * (1 - y);
	}
	
	/**
	# This implements an "exploding" sigmoid activation function
	**/
	public static double ex_sigmoid(double x)
	{
	    return 100 * (sigmoid(x));
	}
	
	/**
	# This implements the tanh activation function
	**/
	public static double tanh(double x)
	{
	    double y = 0;
		try
		{
	        y = Math.tanh(x);
		}
	    catch (Exception e) //Incase of any overflow
		{
	        y = 0;
		}
	
	    return y;
	}
}