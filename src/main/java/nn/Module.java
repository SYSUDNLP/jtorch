package nn;

import java.util.HashMap;
import static nn.TensorMath.*;

// local Module = torch.class('nn.Module')

public class Module
{

// function Module:__init()
//    self.gradInput = torch.Tensor()
//    self.output = torch.Tensor()
// end

	//public HashMap<String, Tensor> parameters = new HashMap<String, Tensor>();

	Tensor _output;
	Tensor _gradInput;

	Tensor _weight;
	Tensor _bias;
	Tensor _gradWeight;
	Tensor _gradBias;

	public Module() {
	}

	public Tensor output() {
		return _output;
	}

	public Tensor gradInput() {
		return _gradInput;
	}

// function Module:parameters()
//    if self.weight and self.bias then
//       return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
//    elseif self.weight then
//       return {self.weight}, {self.gradWeight}
//    elseif self.bias then
//       return {self.bias}, {self.gradBias}
//    else
//       return
//    end
// end

	public java.util.ArrayList<Tensor[]> parameters() {
		java.util.ArrayList<Tensor[]> params = new java.util.ArrayList<Tensor[]>();
		if (_weight != null && _bias != null) {
			java.util.Collections.addAll(params, new Tensor[][] { {_weight, _gradWeight}, {_bias, _gradBias} });
		}
		else if (_weight != null) {
			java.util.Collections.addAll(params, new Tensor[][] { {_weight, _gradWeight} });
		}
		else if (_bias != null) {
			java.util.Collections.addAll(params, new Tensor[][] { {_bias, _gradBias} });
		}
		return params;
	}

// function Module:updateOutput(input)
//    return self.output
// end

	// TO OVERRIDE
	public Tensor updateOutput(Tensor input) {
		return _output;
	}

// function Module:forward(input)
//    return self:updateOutput(input)
// end

	public Tensor forward(Tensor input) {
		return updateOutput(input);
	}

// function Module:backward(input, gradOutput, scale)
//    scale = scale or 1
//    self:updateGradInput(input, gradOutput)
//    self:accGradParameters(input, gradOutput, scale)
//    return self.gradInput
// end

	public Tensor backward(Tensor input, Tensor gradOutput, double scale) {
		updateGradInput(input, gradOutput);
		accGradParameters(input, gradOutput, scale);
		return _gradInput;
	}

	public Tensor backward(Tensor input, Tensor gradOutput) {
		return backward(input, gradOutput, 1);
	}
	
// function Module:backwardUpdate(input, gradOutput, lr)
//    self:updateGradInput(input, gradOutput)
//    self:accUpdateGradParameters(input, gradOutput, lr)
//    return self.gradInput
// end

	public Tensor backwardUpdate(Tensor input, Tensor gradOutput, double lr) {
		updateGradInput(input, gradOutput);
		accUpdateGradParameters(input, gradOutput, lr);
		return _gradInput;
	}

// function Module:updateGradInput(input, gradOutput)
//    return self.gradInput
// end

	// TO OVERRIDE
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		return _gradInput;
	}

// function Module:accGradParameters(input, gradOutput, scale)
// end
	
	// TO OVERRIDE
	public void accGradParameters(Tensor input, Tensor gradOutput, double scale) {
	}

// function Module:accUpdateGradParameters(input, gradOutput, lr)
//    local gradWeight = self.gradWeight
//    local gradBias = self.gradBias
//    self.gradWeight = self.weight
//    self.gradBias = self.bias
//    self:accGradParameters(input, gradOutput, -lr)
//    self.gradWeight = gradWeight
//    self.gradBias = gradBias
// end
	
	public void accUpdateGradParameters(Tensor input, Tensor gradOutput, double lr) {
		/*
		if (_parametersShared) {
			sharedAccUpdateGradParameters(input, gradOutput, lr);
			return;
		}
		*/
		Tensor gradWeight = _gradWeight;
		Tensor gradBias = _gradBias;
		_gradWeight = _weight;
		_gradBias = _bias;
		accGradParameters(input, gradOutput, -lr);
		_gradWeight = gradWeight;
		_gradBias = gradBias;
	}

// function Module:sharedAccUpdateGradParameters(input, gradOutput, lr)
//    if self:parameters() then
//       self:zeroGradParameters()
//       self:accGradParameters(input, gradOutput, 1)
//       self:updateParameters(lr)
//    end
// end
	
	public void sharedAccUpdateGradParameters(Tensor input, Tensor gradOutput, double lr) {
		if (parameters().size() > 0) {
			zeroGradParameters();
			accGradParameters(input, gradOutput, 1);
			updateParameters(lr);
		}
	}

// function Module:zeroGradParameters()
//    local _,gradParams = self:parameters()
//    if gradParams then
//       for i=1,#gradParams do
//          gradParams[i]:zero()
//       end
//    end
// end

	public void zeroGradParameters() {
		for (Tensor[] params : parameters()) params[1].fill(0);
	}
	
// function Module:updateParameters(learningRate)
//    local params, gradParams = self:parameters()
//    if params then
//       for i=1,#params do
//          params[i]:add(-learningRate, gradParams[i])
//       end
//    end
// end
	
	public void updateParameters(double learningRate) {
		for (Tensor[] params : parameters()) {
			Tensor param = params[0];
			Tensor gradParam = params[1];
			add(param, param, -learningRate, gradParam);
		}
	}

// function Module:training()
//    self.train = true
// end

	private boolean train = false;

	public void training() {
		train = true;
	}

// function Module:evaluate()
//    self.train = false
// end

	public void evaluate() {
		train = false;
	}

// function Module:share(mlp, ...)
//    local arg = {...}
//    for i,v in ipairs(arg) do
//       if self[v] ~= nil then
//          self[v]:set(mlp[v])
//          self.accUpdateGradParameters = self.sharedAccUpdateGradParameters
//          mlp.accUpdateGradParameters = mlp.sharedAccUpdateGradParameters
//       end
//    end
//    return self
// end

	/*
	private boolean _parametersShared = false;

	public Module share(Module mlp, String... params) {
		for (String param : params) {
			if (parameters.containsKey(param)) {
				parameters.put(param, new Tensor(mlp.parameters.get(param)));
				_parametersShared = true;
				mlp._parametersShared = true;
			}
		}
		return this;
	}
	*/

// function Module:clone(...)
//    local f = torch.MemoryFile("rw"):binary()
//    f:writeObject(self)
//    f:seek(1)
//    local clone = f:readObject()
//    f:close()
//    if select('#',...) > 0 then
//       clone:share(self,...)
//    end
//    return clone
// end

	public Module clone() {
		throw new RuntimeException("Not implemented.");
	}

// function Module:type(type, tensorCache)
//    assert(type, 'Module: must provide a type to convert to')

//    tensorCache = tensorCache or {}

//    -- find all tensors and convert them
//    for key,param in pairs(self) do
//       self[key] = nn.utils.recursiveType(param, type, tensorCache)
//    end

//    return self
// end

// function Module:float()
//    return self:type('torch.FloatTensor')
// end

// function Module:double()
//    return self:type('torch.DoubleTensor')
// end

// function Module:cuda()
//    return self:type('torch.CudaTensor')
// end

// function Module:reset()
// end

	public Module reset() {
		return this;
	}

// -- This function is not easy to understand. It works as follows:
// --
// -- - gather all parameter tensors for this module (and children);
// --   count all parameter values (floats)
// -- - create one ginormous memory area (Storage object) with room for all
// --   parameters
// -- - remap each parameter tensor to point to an area within the ginormous
// --   Storage, and copy it there
// --
// -- It has the effect of making all parameters point to the same memory area,
// -- which is then returned.
// --
// -- The purpose is to allow operations over all parameters (such as momentum
// -- updates and serialization), but it assumes that all parameters are of
// -- the same type (and, in the case of CUDA, on the same device), which
// -- is not always true. Use for_each() to iterate over this module and
// -- children instead.
// --
// -- Module._flattenTensorBuffer can be used by other packages (e.g. cunn)
// -- to specify the type of temporary buffers. For example, the temporary
// -- buffers for CudaTensor could be FloatTensor, to avoid GPU memory usage.
// --
// -- TODO: This logically belongs to torch.Tensor, not nn.


// Module._flattenTensorBuffer = {}
// function Module.flatten(parameters)

//    -- returns true if tensor occupies a contiguous region of memory (no holes)
//    local function isCompact(tensor)
//       local sortedStride, perm = torch.sort(
//             torch.LongTensor(tensor:nDimension()):set(tensor:stride()), 1, true)
//       local sortedSize = torch.LongTensor(tensor:nDimension()):set(
//             tensor:size()):index(1, perm)
//       local nRealDim = torch.clamp(sortedStride, 0, 1):sum()
//       sortedStride = sortedStride:narrow(1, 1, nRealDim):clone()
//       sortedSize   = sortedSize:narrow(1, 1, nRealDim):clone()
//       local t = tensor.new():set(tensor:storage(), 1,
//                                  sortedSize:storage(),
//                                  sortedStride:storage())
//       return t:isContiguous()
//    end

/*	static boolean _isCompact(Tensor tensor) {
		int[] stride = tensor.stride();
		int[] size = tensor.size();
		// sort in decreasing order
		for (int i = 0; i < stride.length - 1; ++ i) {
			for (int j = stride.length - 1; j > i; -- j) {
				if (stride[j] > stride[j - 1]) {
					int tmp = stride[j];
					stride[j] = stride[j - 1];
					stride[j - 1] = tmp;
					tmp = size[j];
					size[j] = size[j - 1];
					size[j - 1] = tmp;
				}
			}
		}
		// trim dimensions of zero stride
		int nRealDim = 0;
		for (int stride1 : stride) if (stride1 != 0) ++ nRealDim;
		int[] stride2 = new int[nRealDim];
		int[] size2 = new int[nRealDim];
		for (int i = 0, j = 0; i < stride.length; ++ i) {
			if (stride[i] == 0) continue;
			stride2[j] = stride[i];
			size2[j] = size[i];
			++ j;
		}
		// is contiguous
		return Tensor._compareIntArrays(stride2, Tensor._defaultStride(size2));
	}
*/

//    if not parameters or #parameters == 0 then
//       return torch.Tensor()
//    end
//    local Tensor = parameters[1].new
//    local TmpTensor = Module._flattenTensorBuffer[torch.type(parameters[1])] or Tensor

//    -- 1. construct the set of all unique storages referenced by parameter tensors
//    local storages = {}
//    local nParameters = 0
//    local parameterMeta = {}
//    for k = 1,#parameters do
//       local param = parameters[k]
//       local storage = parameters[k]:storage()
//       local storageKey = torch.pointer(storage)

//       if not storages[storageKey] then
//          storages[storageKey] = {storage, nParameters}
//          nParameters = nParameters + storage:size()
//       end

//       parameterMeta[k] = {storageOffset = param:storageOffset() +
//                                           storages[storageKey][2],
//                           size          = param:size(),
//                           stride        = param:stride()}
//    end

//    -- 2. construct a single tensor that will hold all the parameters
//    local flatParameters = TmpTensor(nParameters):zero()

//    -- 3. determine if there are elements in the storage that none of the
//    --    parameter tensors reference ('holes')
//    local tensorsCompact = true
//    for k = 1,#parameters do
//       local meta = parameterMeta[k]
//       local tmp = TmpTensor():set(
//          flatParameters:storage(), meta.storageOffset, meta.size, meta.stride)
//       tmp:fill(1)
//       tensorsCompact = tensorsCompact and isCompact(tmp)
//    end

//    local maskParameters  = flatParameters:byte():clone()
//    local compactOffsets  = flatParameters:long():cumsum(1)
//    local nUsedParameters = compactOffsets[-1]

//    -- 4. copy storages into the flattened parameter tensor
//    for _, storageAndOffset in pairs(storages) do
//       local storage, offset = table.unpack(storageAndOffset)
//       flatParameters[{{offset+1,offset+storage:size()}}]:copy(Tensor():set(storage))
//    end

//    -- 5. allow garbage collection
//    storages = nil
//    for k = 1,#parameters do
//        parameters[k]:set(Tensor())
//    end

//    -- 6. compact the flattened parameters if there were holes
//    if nUsedParameters ~= nParameters then
//       assert(tensorsCompact,
//          "Cannot gather tensors that are not compact")

//       flatParameters = TmpTensor(nUsedParameters):copy(
//             flatParameters:maskedSelect(maskParameters))
//       for k = 1,#parameters do
//         parameterMeta[k].storageOffset =
//               compactOffsets[parameterMeta[k].storageOffset]
//       end
//    end

//    if TmpTensor ~= Tensor then
//       flatParameters = Tensor(flatParameters:nElement()):copy(flatParameters)
//    end

//    -- 7. fix up the parameter tensors to point at the flattened parameters
//    for k = 1,#parameters do
//       parameters[k]:set(flatParameters:storage(),
//           parameterMeta[k].storageOffset,
//           parameterMeta[k].size,
//           parameterMeta[k].stride)
//    end

//    return flatParameters
// end

/*	static Tensor _flatten(Tensor[] parameters) {
		if (parameters == null || parameters.length == 0) return new Tensor();
		throw new RuntimeException("Not supported.");
	}
*/
// function Module:getParameters()
//    -- get parameters
//    local parameters,gradParameters = self:parameters()
//    return Module.flatten(parameters), Module.flatten(gradParameters)
// end

	

// function Module:__call__(input, gradOutput)
//    self:forward(input)
//    if gradOutput then
//       self:backward(input, gradOutput)
//       return self.output, self.gradInput
//    else
//       return self.output
//    end
// end

// -- Run a callback (called with the module as an argument) in preorder over this
// -- module and its children.
// --
// function Module:apply(callback)
//     callback(self)

//     if self.modules then
//         for _, module in ipairs(self.modules) do
//             module:apply(callback)
//         end
//     end
// end

	public void apply(java.util.function.Consumer<Module> consumer) {
		consumer.accept(this);
		if (this instanceof Container) {
			for (Module module : ((Container)this).modules) {
				module.apply(consumer);
			}
		}
	}

// function Module:findModules(typename, container)
//   container = container or self
//   local nodes = {}
//   local containers = {}
//   local mod_type = torch.typename(self)
//   if mod_type == typename then
//     nodes[#nodes+1] = self
//     containers[#containers+1] = container
//   end
//   -- Recurse on nodes with 'modules'
//   if (self.modules ~= nil) then
//     if (torch.type(self.modules) == 'table') then
//       for i = 1, #self.modules do
//         local child = self.modules[i]
//         local cur_nodes, cur_containers =
//           child:findModules(typename, self)
//         assert(#cur_nodes == #cur_containers,
//           'Internal error: incorrect return length')  -- This shouldn't happen
//         -- add the list items from our child to our list (ie return a
//         -- flattened table of the return nodes).
//         for j = 1, #cur_nodes do
//           nodes[#nodes+1] = cur_nodes[j]
//           containers[#containers+1] = cur_containers[j]
//         end
//       end
//     end
//   end
//   return nodes, containers
// end

	public java.util.ArrayList<Module[]> findModules(String typename, Container container) {
		java.util.ArrayList<Module[]> res = new java.util.ArrayList<>();
		String mod_type = this.getClass().getName();
		if (this.getClass().equals(typename)) {
			res.add(new Module[] {this, container});
		}
		if (! (this instanceof Container)) return res;
		for (Module child : ((Container)this).modules) {
			res.addAll(child.findModules(typename, (Container)this));
		}
		return res;
	}

	public java.util.ArrayList<Module[]> findModules(String typename) {
		return findModules(typename, (Container)this);
	}

// -- returns a list of modules
// function Module:listModules()
//    local function tinsert(to, from)
//       if torch.type(from) == 'table' then
//          for i=1,#from do
//             tinsert(to,from[i])
//          end
//       else
//          table.insert(to,from)
//       end
//    end
//    -- include self first
//    local modules = {self}
//    if self.modules then
//       for i=1,#self.modules do
//          local modulas = self.modules[i]:listModules()
//          if modulas then
//             tinsert(modules,modulas)
//          end
//       end
//    end
//    return modules
// end

	public java.util.ArrayList<Module> listModules() {
		java.util.ArrayList<Module> modules = new java.util.ArrayList<>();
		modules.add(this);
		if (this instanceof Container) {
			for (Module child : ((Container)this).modules) {
				modules.addAll(child.listModules());
			}
		}
		return modules;
	}

}