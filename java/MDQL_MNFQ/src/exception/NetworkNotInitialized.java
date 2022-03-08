/**
# This is thrown if the neural network passed to the dql instance
# is not initialized
**/

package exception;

public class NetworkNotInitialized extends Exception{
	private static final long serialVersionUID = 1L;

	public NetworkNotInitialized(String msg)
	{
		super(msg);
	}
}
