/**
# This is thrown if the size of two arrays are not the same that are
# supposed to be
**/

package exception;

public class NotEqualLength extends Exception{
	private static final long serialVersionUID = 1L;

	public NotEqualLength(String msg)
	{
		super(msg);
	}
}
