/**
# This is thrown if the output of the network has not been set
# before propogating forward
**/

package exception;

public class OutputNotSet extends Exception{
	private static final long serialVersionUID = 1L;

	public OutputNotSet(String msg)
	{
		super(msg);
	}
}
