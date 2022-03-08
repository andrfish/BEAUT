/**
# This is thrown if the states or actions passed to the dql class
# are less than 1
**/

package exception;

public class InvalidLength extends Exception{
	private static final long serialVersionUID = 1L;

	public InvalidLength(String msg)
	{
		super(msg);
	}
}
