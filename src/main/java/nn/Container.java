package nn;

// -- This is code common to container modules, which are collections of
// -- smaller constituent modules like Parallel, Sequential, etc.
// local Container, parent = torch.class('nn.Container', 'nn.Module')

public class Container extends Module
{

// function Container:__init(...)
//     parent.__init(self, ...)
//     self.modules = {}
// end

	protected java.util.ArrayList<Module> modules;
	
	public Container() {
		modules = new java.util.ArrayList<>();
	}

// function Container:add(module)
//     table.insert(self.modules, module)
//     return self
// end

	public Container add(Module module) {
		modules.add(module);
		return this;
	}

	public Container insert(Module module, int index) {
		modules.add(index - 1, module);
		return this;
	}

	public Container remove(int index) {
		modules.remove(index - 1);
		return this;
	}

// function Container:get(index)
//     return self.modules[index]
// end

	public Module get(int index) {
		return modules.get(index);
	}

// function Container:size()
//     return #self.modules
// end

	public int size() {
		return modules.size();
	}

// function Container:applyToModules(func)
//     for _, module in ipairs(self.modules) do
//         func(module)
//     end
// end

	public void applyToModules(java.util.function.Consumer<Module> func) {
		for (Module module : modules) func.accept(module);
	}

// function Container:zeroGradParameters()
//     self:applyToModules(function(module) module:zeroGradParameters() end)
// end

	@Override
	public void zeroGradParameters() {
		applyToModules(module -> module.zeroGradParameters());
	}

// function Container:updateParameters(learningRate)
//     self:applyToModules(function(module) module:updateParameters(learningRate) end)
// end

	@Override
	public void updateParameters(double learningRate) {
		applyToModules(module -> module.updateParameters(learningRate));
	}

// function Container:training()
//     self:applyToModules(function(module) module:training() end)
//     parent.training(self)
// end

	@Override
	public void training() {
		applyToModules(module -> module.training());
		super.training();
	}

// function Container:evaluate()
//     self:applyToModules(function(module) module:evaluate() end)
//     parent.evaluate(self)
// end

	@Override
	public void evaluate() {
		applyToModules(module -> module.evaluate());
		super.evaluate();
	}

// function Container:share(mlp, ...)
//     for i=1,#self.modules do
//         self.modules[i]:share(mlp.modules[i], ...);
//     end
// end

	/*
	@Override
	public Module share(Module mlp, String... params) {
		for (Module module : modules) {
			module.share(mlp, params);
		}
		return this;
	}
	*/

// function Container:reset(stdv)
//     self:applyToModules(function(module) module:reset(stdv) end)
// end

	@Override
	public Module reset() {
		applyToModules(module -> module.reset());
		return this;
	}

// function Container:parameters()
//     local function tinsert(to, from)
//         if type(from) == 'table' then
//             for i=1,#from do
//                 tinsert(to,from[i])
//             end
//         else
//             table.insert(to,from)
//         end
//     end
//     local w = {}
//     local gw = {}
//     for i=1,#self.modules do
//         local mw,mgw = self.modules[i]:parameters()
//         if mw then
//             tinsert(w,mw)
//             tinsert(gw,mgw)
//         end
//     end
//     return w,gw
// end

	@Override
	public java.util.ArrayList<Tensor[]> parameters() {
		return parameters(true);
	}

 	public java.util.ArrayList<Tensor[]> parameters(boolean nonduplicated) {
 		java.util.ArrayList<Tensor[]> params = new java.util.ArrayList<>();
 		java.util.HashSet<Tensor> paramSet = new java.util.HashSet<>();
 		for (Module module : modules) {
 			for (Tensor[] param : module.parameters()) {
 				if (nonduplicated && paramSet.contains(param[0])) continue;
 				paramSet.add(param[0]);
 				params.add(param);
 			}
 		}
 		return params;
 	}

}