/**
# This class is used for error functions in my neural network
# code
#
# Author: Andrew Fisher
**/

package neural;

import exception.*;

public class error
{
	/**
	# This implements a mean-squared-error error function
	# TODO: Add support for more types of error functions
	**/
	public static double mse(double[] correct, double[] output) throws NotEqualLength
	{
	    //Check that the lengths are the same
	    if (correct.length != output.length)
	        throw new NotEqualLength("The size of the correct array (" + correct.length + ") is not equal to the size of the output (" + output.length + ")");
	
	    //Calculate the error
	    double ret = 0;
	    for (int i = 0; i < correct.length; i++)
	        ret += Math.pow((correct[i] - output[i]), 2);
	    ret /= correct.length;
	
	    return ret;
	}
}