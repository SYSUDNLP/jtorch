package nn;

// local Threshold, parent = torch.class('nn.Threshold','nn.Module')

public class Threshold extends Module
{

// function Threshold:__init(th,v,ip)
//    parent.__init(self)
//    self.threshold = th or 1e-6
//    self.val = v or 0
//    if (th and type(th) ~= 'number') or (v and type(v) ~= 'number') then
//       error('nn.Threshold(threshold, value)')
//    end
//    -- default for inplace is false
//    self.inplace = ip or false
//    if (ip and type(ip) ~= 'boolean') then
//       error('in-place flag must be boolean')
//    end
//    self:validateParameters()
// end

	double threshold;
	double value;
	boolean inplace;

	public Threshold(double th, double v, boolean ip) {
		threshold = th;
		value = v;
		inplace = ip;
		validateParameters();
	}

	public Threshold() {
		this(1e-6, 0, false);
	}

	@Override
	public Module clone() {
		return new Threshold(threshold, value, inplace);
	}

// function Threshold:updateOutput(input)
//    self:validateParameters()
//    input.nn.Threshold_updateOutput(self, input)
//    return self.output
// end

	@Override
	public Tensor updateOutput(Tensor input) {
		_Threshold_updateOutput(input);
		return _output;
	}

// function Threshold:updateGradInput(input, gradOutput)
//    self:validateParameters()
//    input.nn.Threshold_updateGradInput(self, input, gradOutput)
//    return self.gradInput
// end

	@Override
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		_Threshold_updateGradInput(input, gradOutput);
		return _gradInput;
	}

// function Threshold:validateParameters()
//    self.inplace = self.inplace or false -- backwards compatibility pre inplace
//    if self.inplace then
//       if self.val > self.threshold then
//          error('in-place processing requires value (' .. self.val ..
//                   ') not exceed threshold (' .. self.threshold .. ')')
//       end
//    end
// end

	void validateParameters() {
		if (inplace) {
			assert(value <= threshold);
		}
	}

// #ifndef TH_GENERIC_FILE
// #define TH_GENERIC_FILE "generic/Threshold.c"
// #else

// static int nn_(Threshold_updateOutput)(lua_State *L)
// {
//   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
//   real val = luaT_getfieldchecknumber(L, 1, "val");
//   real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
//   THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
//   int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

//   if (inPlace) {
//     TH_TENSOR_APPLY(real, input,                   \
//                     if (*input_data <= threshold) { \
//                       *input_data = val;           \
//                     });
//     THTensor_(set)(output, input);
//   } else {
//     THTensor_(resizeAs)(output, input);
//     TH_TENSOR_APPLY2(real, output, real, input,                         \
//                      *output_data = (*input_data > threshold) ? *input_data : val;);

//   }

//   return 1;
// }

	void _Threshold_updateOutput(Tensor input) {
		if (inplace) {
			input.apply( x -> (x <= threshold ? value : x) );
			_output = new Tensor(input);
		}
		else {
			_output = new Tensor(input.size());
			_output.map(input, x -> (x <= threshold ? value : x) );
		}
	}

// static int nn_(Threshold_updateGradInput)(lua_State *L)
// {
//   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
//   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
//   real threshold = luaT_getfieldchecknumber(L, 1, "threshold");
//   THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
//   int inPlace = luaT_getfieldcheckboolean(L, 1, "inplace");

//   if (inPlace) {
//     TH_TENSOR_APPLY2(real, gradOutput, real, input,    \
//                      if ((*input_data) <= threshold) { \
//                        *gradOutput_data = 0;           \
//                          });
//     THTensor_(set)(gradInput, gradOutput);
//   } else {
//     THTensor_(resizeAs)(gradInput, input);
//     TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, input,    \
//                      if ((*input_data) > threshold) *gradInput_data = *gradOutput_data; \
//                      else *gradInput_data = 0;);                        \
//   }

//   return 1;
// }

	void _Threshold_updateGradInput(Tensor input, Tensor gradOutput) {
		if (inplace) {
			gradOutput.map(input, (gradOut, in) -> (in <= threshold ? 0 : gradOut) );
			_gradInput = new Tensor(gradOutput);
		}
		else {
			_gradInput = new Tensor(input.size());
			_gradInput.map2(gradOutput, input, (gradOut, in) -> (in <= threshold ? 0 : gradOut) );
		}
	}

// static const struct luaL_Reg nn_(Threshold__) [] = {
//   {"Threshold_updateOutput", nn_(Threshold_updateOutput)},
//   {"Threshold_updateGradInput", nn_(Threshold_updateGradInput)},
//   {NULL, NULL}
// };

// static void nn_(Threshold_init)(lua_State *L)
// {
//   luaT_pushmetatable(L, torch_Tensor);
//   luaT_registeratname(L, nn_(Threshold__), "nn");
//   lua_pop(L,1);
// }

// #endif
}