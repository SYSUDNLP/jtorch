package nn;

// local ReLU, Parent = torch.class('nn.ReLU', 'nn.Threshold')

public class ReLU extends Threshold
{

// function ReLU:__init(p)
//    Parent.__init(self,0,0,p)
// end

	public ReLU(boolean inplace) {
		super(0, 0, inplace);
	}

	public ReLU() {
		this(false);
	}

	@Override
	public Module clone() {
		return new ReLU(inplace);
	}

}