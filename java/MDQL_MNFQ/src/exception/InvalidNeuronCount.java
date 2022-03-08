/**
# This is thrown if the number of neurons specified for a layer
# is less than or equal to zero
**/

package exception;

public class InvalidNeuronCount extends Exception{
	private static final long serialVersionUID = 1L;

	public InvalidNeuronCount(String msg)
	{
		super(msg);
	}
}
