
package nn;

import static nn.TensorMath.*;

// local Tanh = torch.class('nn.Tanh', 'nn.Module')

public class Tanh extends Module
{

	@Override
	public Module clone() {
		return new Tanh();
	}

// function Tanh:updateOutput(input)
//    return input.nn.Tanh_updateOutput(self, input)
// end

	@Override
	public Tensor updateOutput(Tensor input) {
		return _Tanh_updateOutput(input);
	}


// function Tanh:updateGradInput(input, gradOutput)
//    return input.nn.Tanh_updateGradInput(self, input, gradOutput)
// end

	@Override
	public Tensor updateGradInput(Tensor input, Tensor gradOutput) {
		return _Tanh_updateGradInput(input, gradOutput);
	}


// #ifndef TH_GENERIC_FILE
// #define TH_GENERIC_FILE "generic/Tanh.c"
// #else

// static int nn_(Tanh_updateOutput)(lua_State *L)
// {
//   THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
//   THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

//   THTensor_(resizeAs)(output, input);
//   THTensor_(tanh)(output, input);
//   return 1;
// }

	Tensor _Tanh_updateOutput(Tensor input) {
		_output = new Tensor(input.size());
		_output.map(input, x -> Math.tanh(x) );
		return _output;
	}

// static int nn_(Tanh_updateGradInput)(lua_State *L)
// {
//   THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
//   THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
//   THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

//   THTensor_(resizeAs)(gradInput, output);

//   if (output->nDimension == 1 || 
//       !THTensor_(isContiguous)(output) || 
//       !THTensor_(isContiguous)(gradOutput) ||
//       !THTensor_(isContiguous)(gradInput))
//   {
//     TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,  \
//          real z = *output_data;            \
//          *gradInput_data = *gradOutput_data * (1. - z*z););
//   }
//   else
//   {
//     real* ptr_gradOutput = THTensor_(data)(gradOutput);
//     real* ptr_gradInput  = THTensor_(data)(gradInput);
//     real* ptr_output     = THTensor_(data)(output);
//     long i;

// #pragma omp parallel for private(i)
//     for(i = 0; i < THTensor_(nElement)(gradInput); i++)
//     {
//       real z = ptr_output[i];
//       ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
//     }
//   }
//   return 1;
// }

	Tensor _Tanh_updateGradInput(Tensor input, Tensor gradOutput) {
		_gradInput = new Tensor(input.size());
		_gradInput.map2(gradOutput, _output, (g, y) -> g * (1 - y * y) );
		return _gradInput;
	}

// static const struct luaL_Reg nn_(Tanh__) [] = {
//   {"Tanh_updateOutput", nn_(Tanh_updateOutput)},
//   {"Tanh_updateGradInput", nn_(Tanh_updateGradInput)},
//   {NULL, NULL}
// };

// static void nn_(Tanh_init)(lua_State *L)
// {
//   luaT_pushmetatable(L, torch_Tensor);
//   luaT_registeratname(L, nn_(Tanh__), "nn");
//   lua_pop(L,1);

// }

// #endif

}