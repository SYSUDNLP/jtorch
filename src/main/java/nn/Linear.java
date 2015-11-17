package nn;

// local Linear, parent = torch.class('nn.Linear', 'nn.Module')


import static nn.TensorMath.*;

public class Linear extends Module
{

// function Linear:__init(inputSize, outputSize)
//    parent.__init(self)

//    self.weight = torch.Tensor(outputSize, inputSize)
//    self.bias = torch.Tensor(outputSize)
//    self.gradWeight = torch.Tensor(outputSize, inputSize)
//    self.gradBias = torch.Tensor(outputSize)

//    self:reset()
// end

	public Linear(int inputSize, int outputSize, boolean updateGradInput) {
		_weight = new Tensor(outputSize, inputSize);
		_gradWeight = new Tensor(_weight.size());
		_bias = new Tensor(outputSize);
		_gradBias = new Tensor(_bias.size());
		reset();

		if (updateGradInput) {
			_gradInput = new Tensor();
		}
	}

	public Linear(int inputSize, int outputSize) {
		this(inputSize, outputSize, true);
	}

	void _copy(Linear linear, boolean shared) {
		if (shared) {
			_weight = linear._weight;
			_gradWeight = linear._gradWeight;
			_bias = linear._bias;
			_gradBias = linear._gradBias;
		}
		else {
			_weight = linear._weight.clone();
			_gradWeight = linear._gradWeight.clone();
			_bias = linear._bias.clone();
			_gradBias = linear._gradBias.clone();
		}
		if (linear._gradInput != null) {
			_gradInput = new Tensor();
		}
	}

	public Linear(Linear shared) {
		_copy(shared, true);
	}

	@Override
	public Module clone() {
		Linear clone = new Linear(_weight.size(2), _weight.size(1), _gradInput != null);
		clone._copy(this, false);
		return clone;
	}

// function Linear:reset(stdv)
//    if stdv then
//       stdv = stdv * math.sqrt(3)
//    else
//       stdv = 1./math.sqrt(self.weight:size(2))
//    end
//    if nn.oldSeed then
//       for i=1,self.weight:size(1) do
//          self.weight:select(1, i):apply(function()
//             return torch.uniform(-stdv, stdv)
//          end)
//          self.bias[i] = torch.uniform(-stdv, stdv)
//       end
//    else
//       self.weight:uniform(-stdv, stdv)
//       self.bias:uniform(-stdv, stdv)
//    end

//    return self
// end

	public Module reset(double stdv) {
		stdv = stdv * Math.sqrt(3);
		TensorMath.uniform(_weight, -stdv, stdv, _weight.size());
		TensorMath.uniform(_bias, -stdv, stdv, _bias.size());
		return this;
	}

	@Override
	public Module reset() {
		double stdv = 1.0 / Math.sqrt(_weight.size(2));
		return reset(stdv);
	}

// function Linear:updateOutput(input)
//    if input:dim() == 1 then
//       self.output:resize(self.bias:size(1))
//       self.output:copy(self.bias)
//       self.output:addmv(1, self.weight, input)
//    elseif input:dim() == 2 then
//       local nframe = input:size(1)
//       local nElement = self.output:nElement()
//       self.output:resize(nframe, self.bias:size(1))
//       if self.output:nElement() ~= nElement then
//          self.output:zero()
//       end
//       self.addBuffer = self.addBuffer or input.new()
//       if self.addBuffer:nElement() ~= nframe then
//          self.addBuffer:resize(nframe):fill(1)
//       end
//       self.output:addmm(0, self.output, 1, input, self.weight:t())
//       self.output:addr(1, self.addBuffer, self.bias)
//    else
//       error('input must be vector or matrix')
//    end

//    return self.output
// end

	// this implementation is different than the torch implementatin, which is confusing.
	@Override
	public Tensor updateOutput(Tensor input) {
		if (input.dim() == 1) {
			_output = new Tensor(_bias.size(1));
			add(_output, _bias, ParallelTasks.mv(_weight, input));
		}
		else if (input.dim() == 2) {
			int nframe = input.size(1);
			_output = ParallelTasks.mm(input, _weight.t());
			add(_output, _output, _bias.view(1, _bias.size(1)).expand(_output.size()));
		}
		else {
			throw new IllegalArgumentException("Input must be vector or matrix.");
		}
		return _output;
	}

// function Linear:updateGradInput(input, gradOutput)
//    if self.gradInput then

//       local nElement = self.gradInput:nElement()
//       self.gradInput:resizeAs(input)
//       if self.gradInput:nElement() ~= nElement then
//          self.gradInput:zero()
//       end
//       if input:dim() == 1 then
//          self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
//       elseif input:dim() == 2 then
//          self.gradInput:addmm(0, 1, gradOutput, self.weight)
//       end

//       return self.gradInput
//    end
// end

	// this implementation is different than the torch implementatin, which is confusing.
	@Override
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		if (_gradInput != null) {
			if (input.dim() == 1) {
				_gradInput = ParallelTasks.mv(_weight.t(), gradOutput);
			}
			else {
				_gradInput = ParallelTasks.mm(gradOutput, _weight);
			}
		}
		return _gradInput;
	}

// function Linear:accGradParameters(input, gradOutput, scale)
//    scale = scale or 1
//    if input:dim() == 1 then
//       self.gradWeight:addr(scale, gradOutput, input)
//       self.gradBias:add(scale, gradOutput)
//    elseif input:dim() == 2 then
//       self.gradWeight:addmm(scale, gradOutput:t(), input)
//       self.gradBias:addmv(scale, gradOutput:t(), self.addBuffer)
//    end
// end

	@Override
	public void accGradParameters(Tensor input, Tensor gradOutput, double scale) {
		if (input.dim() == 1) {
			add(_gradWeight, _gradWeight, scale, ger(gradOutput, input));
			add(_gradBias, _gradBias, scale, gradOutput);
		}
		else if (input.dim() == 2) {
			add(_gradWeight, _gradWeight, scale, ParallelTasks.mm(gradOutput.t(), input));
			add(_gradBias, _gradBias, scale, sum(gradOutput, 1).view(_gradBias.size()));
		}
	}

	public void accGradParameters(Tensor input, Tensor gradOutput) {
		accGradParameters(input, gradOutput, 1);
	}

// -- we do not need to accumulate parameters when sharing
// Linear.sharedAccUpdateGradParameters = Linear.accUpdateGradParameters

	/*
	public void sharedAccUpdateGradParameters(Tensor input, Tensor gradOutput, double scale) {
		accUpdateGradParameters(input, gradOutput, scale);
	}
	*/

// function Linear:__tostring__()
//   return torch.type(self) ..
//       string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
// end

	@Override
	public String toString() {
		return getClass().getName() + "(" + _weight.size(2) + " -> " + _weight.size(1) + ")";
	}

}