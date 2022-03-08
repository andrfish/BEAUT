/**
# This is thrown if the learning rate is not passed to the layer
# class when creating it
**/

package exception;

public class NoLearningRate extends Exception{
	private static final long serialVersionUID = 1L;

	public NoLearningRate(String msg)
	{
		super(msg);
	}
}
