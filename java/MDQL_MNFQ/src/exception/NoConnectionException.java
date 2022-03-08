/**
# This is thrown when there isn't a next layer and you try to go 
# forward or backward
**/

package exception;

public class NoConnectionException extends Exception{
	private static final long serialVersionUID = 1L;

	public NoConnectionException(String msg)
	{
		super(msg);
	}
}
