/**
# This is thrown if the neural network passed to the dql instance
# is not initialized
**/

package exception;

public class NoFilenameSpecified extends Exception{
	private static final long serialVersionUID = 1L;

	public NoFilenameSpecified(String msg)
	{
		super(msg);
	}
}
