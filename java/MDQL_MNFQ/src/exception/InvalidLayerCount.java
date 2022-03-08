/**
# This is thrown when the size doesn't match the number of neurons
# or is too small for a network
**/

package exception;

public class InvalidLayerCount extends Exception{
	private static final long serialVersionUID = 1L;

	public InvalidLayerCount(String msg)
	{
		super(msg);
	}
}
