package nn;

// local Sequential, _ = torch.class('nn.Sequential', 'nn.Container')

public class Sequential extends Container
{

// function Sequential:__len()
//    return #self.modules
// end

	public Tensor output() {
		return modules.get(modules.size() - 1).output();
	}

	@Override
	public Tensor gradInput() {
		return modules.get(0).gradInput();
	}

	@Override
	public Module clone() {
		Sequential clone = new Sequential();
		for (int i = 1; i <= size(); ++ i) {
			clone.add(get(i).clone());
		}
		return clone;
	}

// function Sequential:add(module)
//    if #self.modules == 0 then
//       self.gradInput = module.gradInput
//    end
//    table.insert(self.modules, module)
//    self.output = module.output
//    return self
// end

	@Override
	public Container add(Module module) {
		return insert(module, modules.size() + 1);
	}

// function Sequential:insert(module, index)
//    index = index or (#self.modules + 1)
//    if index > (#self.modules + 1) or index < 1 then
//       error"index should be contiguous to existing modules"
//    end
//    table.insert(self.modules, index, module)
//    self.output = self.modules[#self.modules].output
//    self.gradInput = self.modules[1].gradInput
// end

	@Override
	public Container insert(Module module, int index) {
		modules.add(index - 1, module);
		return this;
	}

// function Sequential:remove(index)
//    index = index or #self.modules
//    if index > #self.modules or index < 1 then
//       error"index out of range"
//    end
//    table.remove(self.modules, index)
//    if #self.modules > 0 then
//        self.output = self.modules[#self.modules].output
//        self.gradInput = self.modules[1].gradInput
//    else
//        self.output = torch.Tensor()
//        self.gradInput = torch.Tensor()
//    end
// end

	@Override
	public Container remove(int index) {
		modules.remove(index - 1);
		return this;
	}

// function Sequential:updateOutput(input)
//    local currentOutput = input
//    for i=1,#self.modules do
//       currentOutput = self.modules[i]:updateOutput(currentOutput)
//    end
//    self.output = currentOutput
//    return currentOutput
// end

	@Override
	public Tensor updateOutput(Tensor input) {
		for (Module module : modules) {
			input = module.updateOutput(input);
		}
		//output = input;
		return output();
	}


// function Sequential:updateGradInput(input, gradOutput)
//    local currentGradOutput = gradOutput
//    local currentModule = self.modules[#self.modules]
//    for i=#self.modules-1,1,-1 do
//       local previousModule = self.modules[i]
//       currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
//       currentModule = previousModule
//    end
//    currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
//    self.gradInput = currentGradOutput
//    return currentGradOutput
// end

	@FunctionalInterface
	interface Consumer3<T1, T2, T3>
	{
		public void accept(T1 t1, T2 t2, T3 t3);
	}

	protected void _applyBackward(Tensor input, Tensor gradOutput, Consumer3<Integer, Tensor, Tensor> consumer3) {
		for (int i = modules.size() - 1; i >= 0; -- i) {
			Tensor currentInput = (i == 0 ? input : modules.get(i - 1).output());
			Tensor currentGradOutput = (i == modules.size() - 1 ? gradOutput : modules.get(i + 1).gradInput());
			consumer3.accept(i, currentInput, currentGradOutput);
		}
	}

	@Override
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		_applyBackward(input, gradOutput, (i, curInput, curGradOutput) -> {
			modules.get(i).updateGradInput(curInput, curGradOutput);
		});
		return gradInput();
	}

// function Sequential:accGradParameters(input, gradOutput, scale)
//    scale = scale or 1

//    local currentGradOutput = gradOutput
//    local currentModule = self.modules[#self.modules]
//    for i=#self.modules-1,1,-1 do
//       local previousModule = self.modules[i]
//       currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
//       currentGradOutput = currentModule.gradInput
//       currentModule = previousModule
//    end

//    currentModule:accGradParameters(input, currentGradOutput, scale)
// end

	@Override
	public void accGradParameters(Tensor input, Tensor gradOutput, double scale) {
		_applyBackward(input, gradOutput, (i, curInput, curGradOutput) -> {
			modules.get(i).accGradParameters(curInput, curGradOutput, scale);
		});
	}

// function Sequential:backward(input, gradOutput, scale)
//    scale = scale or 1
//    local currentGradOutput = gradOutput
//    local currentModule = self.modules[#self.modules]
//    for i=#self.modules-1,1,-1 do
//       local previousModule = self.modules[i]
//       currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
//       currentModule.gradInput = currentGradOutput
//       currentModule = previousModule
//    end
//    currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
//    self.gradInput = currentGradOutput
//    return currentGradOutput
// end

	@Override
	public Tensor backward(Tensor input, Tensor gradOutput, double scale) {
		_applyBackward(input, gradOutput, (i, curInput, curGradOutput) -> {
			modules.get(i).backward(curInput, curGradOutput, scale);
		});
		return gradInput();
	}

// function Sequential:accUpdateGradParameters(input, gradOutput, lr)
//    local currentGradOutput = gradOutput
//    local currentModule = self.modules[#self.modules]
//    for i=#self.modules-1,1,-1 do
//       local previousModule = self.modules[i]
//       currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
//       currentGradOutput = currentModule.gradInput
//       currentModule = previousModule
//    end

//    currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
// end

	@Override
	public void accUpdateGradParameters(Tensor input, Tensor gradOutput, double lr) {
		_applyBackward(input, gradOutput, (i, curInput, curGradOutput) -> {
			modules.get(i).accUpdateGradParameters(curInput, curGradOutput, lr);
		});
	}

// function Sequential:__tostring__()
//    local tab = '  '
//    local line = '\n'
//    local next = ' -> '
//    local str = 'nn.Sequential'
//    str = str .. ' {' .. line .. tab .. '[input'
//    for i=1,#self.modules do
//       str = str .. next .. '(' .. i .. ')'
//    end
//    str = str .. next .. 'output]'
//    for i=1,#self.modules do
//       str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
//    end
//    str = str .. line .. '}'
//    return str
// end

	@Override
	public String toString() {
		String tab = "  ";
		String line = "\n";
		String next = " -> ";
		StringBuffer buf = new StringBuffer();
		buf.append(getClass().getName());
		buf.append(" {" + line + tab + "[input");
		for (int i = 0; i < modules.size(); ++ i) {
			buf.append(next + "(" + (i + 1) + ")");
		}
		buf.append( next + "output]");
		for (int i = 0; i < modules.size(); ++ i) {
			buf.append(line + tab + "(" + (i + 1) + "): " + modules.get(i).toString().replaceAll(line, line + tab));
		}
		buf.append(line + "}");
		return buf.toString();
	}

}