package nn;

// local Concat, parent = torch.class('nn.Concat', 'nn.Container')

public class Concat extends Container
{
// function Concat:__init(dimension)
//    parent.__init(self)
//    self.size = torch.LongStorage()
//    self.dimension = dimension
// end

	int outputDimension;
	int[] size;

	public Concat(int outputDimension) {
		this.outputDimension = outputDimension;
	}

	@Override
	public Module clone() {
		Concat clone = new Concat(outputDimension);
		for (int i = 1; i <= size(); ++ i) {
			clone.add(get(i).clone());
		}
		return clone;
	}

// function Concat:updateOutput(input)
//    local outs = {}
//    for i=1,#self.modules do
//       local currentOutput = self.modules[i]:updateOutput(input)
//       outs[i] = currentOutput
//       if i == 1 then
//          self.size:resize(currentOutput:dim()):copy(currentOutput:size())
//       else
//          self.size[self.dimension] = self.size[self.dimension] + currentOutput:size(self.dimension)
//       end
//    end
//    self.output:resize(self.size)

//    local offset = 1
//    for i,module in ipairs(self.modules) do
//       local currentOutput = outs[i]
//       self.output:narrow(self.dimension, offset, currentOutput:size(self.dimension)):copy(currentOutput)
//       offset = offset + currentOutput:size(self.dimension)
//    end
//    return self.output
// end

	Tensor _getInput(Tensor input, int index) {
		return input;
	}

	Tensor _getOutput(Tensor gradOutput, Tensor currentOutput, int offset) {
		int outputSize = currentOutput.size(outputDimension);
		return gradOutput.narrow(outputDimension, offset, outputSize);
	}

	void _updateSize(int i, Tensor currentOutput) {
		if (i == 1) {
			size = currentOutput.size();
		}
		else {
			size[outputDimension - 1] += currentOutput.size(outputDimension);
		}
	}

	@Override
	public Tensor updateOutput(Tensor input) {
		java.util.ArrayList<Tensor> outputs = new java.util.ArrayList<>();
	
		for (int i = 1; i <= modules.size(); ++ i) {
			Tensor currentInput = _getInput(input, i);
			Tensor currentOutput = modules.get(i - 1).updateOutput(currentInput);
			outputs.add(currentOutput);
			_updateSize(i, currentOutput);
		}

		_output = new Tensor(size);
		int offset = 1;
		for (int i = 1; i <= modules.size(); ++ i) {
			Tensor currentOutput = outputs.get(i - 1);
			Tensor output = _getOutput(_output, currentOutput, offset);
			output.copy(currentOutput);
			offset += currentOutput.size(outputDimension);
		}
		return _output;
	}

// function Concat:updateGradInput(input, gradOutput)
//    self.gradInput:resizeAs(input)

//    local offset = 1
//    for i,module in ipairs(self.modules) do
//       local currentOutput = module.output
//       local currentGradInput = module:updateGradInput(input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)))

//       if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
//          if i==1 then
//             self.gradInput:copy(currentGradInput)
//          else
//             self.gradInput:add(currentGradInput)
//          end
//       end
//       offset = offset + currentOutput:size(self.dimension)
//    end
//    return self.gradInput
// end

	@FunctionalInterface
	interface Consumer3<T1, T2, T3>
	{
		public void accept(T1 t1, T2 t2, T3 t3);
	}

	void _applyBackward(Tensor input, Tensor gradOutput, Consumer3<Integer, Tensor, Tensor> consumer3) {
		int offset = 1;
		for (int i = 1; i <= modules.size(); ++ i) {
			Tensor currentInput = _getInput(input, i);
			Tensor currentOutput = modules.get(i - 1).output();
			Tensor currentGradOutput = _getOutput(gradOutput, currentOutput, offset);

			consumer3.accept(i, currentInput, currentGradOutput);

			offset += currentOutput.size(outputDimension);
		}
	}

	@Override
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		_gradInput = new Tensor(input.size());

		_applyBackward(input, gradOutput, (i, currentInput, currentGradOutput) -> {
			Tensor currentGradInput = modules.get(i - 1).updateGradInput(currentInput, currentGradOutput);
			if (currentGradInput != null) {
				TensorMath.add(_gradInput, _gradInput, currentGradInput);				
			}
		});

		return _gradInput;
	}

// function Concat:accGradParameters(input, gradOutput, scale)
//    scale = scale or 1
//    local offset = 1
//    for i,module in ipairs(self.modules) do
//       local currentOutput = module.output
//       module:accGradParameters(
//           input,
//           gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
//           scale)
//       offset = offset + currentOutput:size(self.dimension)
//    end
// end

	@Override
	public void accGradParameters(Tensor input, Tensor gradOutput, double scale) {
		_applyBackward(input, gradOutput, (i, currentInput, currentGradOutput) -> {
			modules.get(i - 1).accGradParameters(currentInput, currentGradOutput, scale);
		});
	}

// function Concat:backward(input, gradOutput, scale)
//    self.gradInput:resizeAs(input)
//    scale = scale or 1
//    local offset = 1
//    for i,module in ipairs(self.modules) do
//       local currentOutput = module.output
//       local currentGradInput = module:backward(input, gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)), scale)
//       if currentGradInput then -- if the module does not produce a gradInput (for example first layer), then ignore it and move on.
//          if i==1 then
//             self.gradInput:copy(currentGradInput)
//          else
//             self.gradInput:add(currentGradInput)
//          end
//       end
//       offset = offset + currentOutput:size(self.dimension)
//    end
//    return self.gradInput
// end

	@Override
	public Tensor backward(Tensor input, Tensor gradOutput, double scale) {
		_gradInput = new Tensor(input.size());

		_applyBackward(input, gradOutput, (i, currentInput, currentGradOutput) -> {
			Tensor currentGradInput = modules.get(i - 1).backward(currentInput, currentGradOutput, scale);
			if (currentGradInput != null) {
				TensorMath.add(_gradInput, _gradInput, currentGradInput);				
			}
		});

		return _gradInput;
	}

// function Concat:accUpdateGradParameters(input, gradOutput, lr)
//    local offset = 1
//    for i,module in ipairs(self.modules) do
//       local currentOutput = module.output
//       module:accUpdateGradParameters(
//           input,
//           gradOutput:narrow(self.dimension, offset, currentOutput:size(self.dimension)),
//           lr)
//       offset = offset + currentOutput:size(self.dimension)
//    end
// end

	@Override
	public void accUpdateGradParameters(Tensor input, Tensor gradOutput, double lr) {
		_applyBackward(input, gradOutput, (i, currentInput, currentGradOutput) -> {
			modules.get(i - 1).accUpdateGradParameters(currentInput, currentGradOutput, lr);
		});
	}

// function Concat:__tostring__()
//    local tab = '  '
//    local line = '\n'
//    local next = '  |`-> '
//    local ext = '  |    '
//    local extlast = '       '
//    local last = '   ... -> '
//    local str = torch.type(self)
//    str = str .. ' {' .. line .. tab .. 'input'
//    for i=1,#self.modules do
//       if i == self.modules then
//          str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. extlast)
//       else
//          str = str .. line .. tab .. next .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab .. ext)
//       end
//    end
//    str = str .. line .. tab .. last .. 'output'
//    str = str .. line .. '}'
//    return str
// end

	@Override
	public String toString() {
		String tab = "  ";
		String line = "\n";
		String next = "  |`-> ";
		String ext = "  |    ";
		String extlast = "       ";
		String last = "   ... -> ";
		StringBuffer buf = new StringBuffer();
		buf.append(getClass().getName());
		buf.append(" {" + line + tab + "input");
		for (int i = 0; i < modules.size(); ++ i) {
			buf.append(line + tab + next + "(" + (i + 1) + "): " + modules.get(i).toString().replaceAll(line, line + tab + (i == modules.size() - 1 ? extlast : ext)));
		}
		buf.append(line + tab + last + "output" + line + "}");
		return buf.toString();
	}

}