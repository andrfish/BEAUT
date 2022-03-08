/**
# This is thrown if an argument that was expected to be passed is
# not passed
**/

package exception;

public class MissingArgument extends Exception{
	private static final long serialVersionUID = 1L;

	public MissingArgument(String msg)
	{
		super(msg);
	}
}
