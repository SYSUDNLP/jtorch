/*
This program contains test-cases and examples for 
	Jacobian.java

*/

package nn;

import static nn.TensorMath.*;

public final class Test1
{

	static void print(int[] array) {
		System.out.print("[");
		for (int i = 0; i < array.length; ++ i) {
			if (i != 0) System.out.print(",");
			System.out.print(array[i]);
		}
		System.out.print("]\n");
	}

	static void print(Tensor t) {
		System.out.print(t);
	}

	static void print(Tensor[] ts) {
		System.out.println("Array of " + ts.length + " tensors:\n");
		for (Tensor t : ts) print(t);
	}

	static void print(String text) {
		System.out.println(text);
	}

	static long time1 = System.currentTimeMillis();
	
	static void tic() {
		time1 = System.currentTimeMillis();
	}

	static void toc() {
		long elapsed = System.currentTimeMillis() - time1;
		java.text.SimpleDateFormat formater = new java.text.SimpleDateFormat("mm:ss:ms");
		System.out.println("Time elapsed: " + formater.format(new java.util.Date(elapsed)));
		tic();
	}

	///////////////////////////

	static void testJacobian(Module module, Tensor input) {
		java.util.ArrayList<Tensor[]> params = module.parameters();
		print("#params = " + params.size());
		Tensor errors = new Tensor(1 + 2 * params.size());
		// get max error on gradInput
		int count = 0;
		errors.set(Jacobian.testJacobian(module, input), ++ count);
		//*
		// get max error on gradWeight
		for (int i = 1; i <= params.size(); ++ i) {
			errors.set(Jacobian.testJacobianParameters(module, input, params.get(i - 1)[0], params.get(i - 1)[1]), ++ count);
		}
		// get max error on gradient update
		for (int i = 1; i <= params.size(); ++ i) {
			errors.set(Jacobian.testJacobianUpdateParameters(module, input, params.get(i - 1)[0]), ++ count);
		}
		/**/
		System.out.println("max(errors) = " + max(errors));
	}

	/*
	jtorch.Sequential {
	  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> output]
	  (1): jtorch.Linear(20 -> 30)
	  (2): jtorch.Sigmoid@6ce253f1
	  (3): jtorch.Linear(30 -> 40)
	  (4): jtorch.Tanh@53d8d10a
	  (5): jtorch.Linear(40 -> 50)
	  (6): jtorch.ReLU@e9e54c2
	  (7): jtorch.Linear(50 -> 60)
	}	
	#params = 8
	max(errors) = 5.018459114936036E-8
	Time elapsed: 01:13:113		
	*/	
	static void testJacobian1() {
		Sequential mlp = new Sequential();
		mlp.add(new Linear(20,30));
		mlp.add(new Sigmoid());
		mlp.add(new Linear(30,40));
		mlp.add(new Tanh());
		mlp.add(new Linear(40,50));
		mlp.add(new ReLU());
		mlp.add(new Linear(50,60));
		System.out.println(mlp);
		testJacobian(mlp, zeros(1, 20));	
	}

	/*
	jtorch.Sequential {
	  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> output]
	  (1): jtorch.Linear(10 -> 10)
	  (2): jtorch.Sigmoid@6ce253f1
	  (3): jtorch.Linear(10 -> 10)
	  (4): jtorch.Sigmoid@53d8d10a
	  (5): jtorch.Linear(10 -> 10)
	  (6): jtorch.Tanh@e9e54c2
	  (7): jtorch.Linear(10 -> 10)
	  (8): jtorch.Tanh@65ab7765
	  (9): jtorch.Linear(10 -> 10)
	  (10): jtorch.ReLU@1b28cdfa
	  (11): jtorch.Linear(10 -> 10)
	  (12): jtorch.ReLU@eed1f14
	  (13): jtorch.Linear(10 -> 10)
	}
	#params = 2
	max(errors) = 4.525722863135684E-8
	Time elapsed: 00:07:07
	*/
	static void testJacobian2() {
		Sequential mlp = new Sequential();
		Linear shared = new Linear(10,10);
		mlp.add(new Linear(shared));
		mlp.add(new Sigmoid());
		mlp.add(new Linear(shared));
		mlp.add(new Sigmoid());
		mlp.add(new Linear(shared));
		mlp.add(new Tanh());
		mlp.add(new Linear(shared));
		mlp.add(new Tanh());
		mlp.add(new Linear(shared));
		mlp.add(new ReLU());
		mlp.add(new Linear(shared));
		mlp.add(new ReLU());
		mlp.add(new Linear(shared));
		System.out.println(mlp);
		testJacobian(mlp, zeros(10, 10));		
	}

	/*
	jtorch.Concat {
	  input
	    |`-> (1): jtorch.Linear(10 -> 10)
	    |`-> (2): jtorch.Sigmoid@6ce253f1
	    |`-> (3): jtorch.Linear(10 -> 10)
	    |`-> (4): jtorch.Tanh@53d8d10a
	    |`-> (5): jtorch.Linear(10 -> 10)
	    |`-> (6): jtorch.ReLU@e9e54c2
	    |`-> (7): jtorch.Linear(10 -> 10)
	     ... -> output
	}
	[1,70]
	[5,70]
	#params = 8
	max(errors) = 1.5541337106128594E-9
	Time elapsed: 00:33:033
	*/
	static void testJacobian3() {
		Concat mlp = new Concat(2);
		mlp.add(new Linear(10,10));
		mlp.add(new Sigmoid());
		mlp.add(new Linear(10,10));
		mlp.add(new Tanh());
		mlp.add(new Linear(10,10));
		mlp.add(new ReLU());
		mlp.add(new Linear(10,10));
		System.out.println(mlp);
		print(mlp.forward(ones(1,10)).size());
		print(mlp.forward(ones(5,10)).size());
		testJacobian(mlp, zeros(5,10));
	}

	/*
	jtorch.DepthConcat {
	  input
	    |`-> (1): jtorch.Linear(10 -> 6)
	    |`-> (2): jtorch.Sigmoid@6ce253f1
	    |`-> (3): jtorch.Linear(10 -> 8)
	    |`-> (4): jtorch.Tanh@53d8d10a
	    |`-> (5): jtorch.Linear(10 -> 10)
	    |`-> (6): jtorch.ReLU@e9e54c2
	    |`-> (7): jtorch.Linear(10 -> 12)
	     ... -> output
	}
	[7,12]
	[35,12]
	#params = 8
	max(errors) = 1.5797376740067648E-9
	Time elapsed: 00:39:039
	*/
	static void testJacobian4() {
		DepthConcat mlp = new DepthConcat(1);
		mlp.add(new Linear(10,6));
		mlp.add(new Sigmoid());
		mlp.add(new Linear(10,8));
		mlp.add(new Tanh());
		mlp.add(new Linear(10,10));
		mlp.add(new ReLU());
		mlp.add(new Linear(10,12));
		System.out.println(mlp);
		print(mlp.forward(ones(1,10)).size());
		print(mlp.forward(ones(5,10)).size());
		testJacobian(mlp, zeros(5,10));
	}


	static void testStochasticGradient1() {
		/*
		dataset={};
		function dataset:size() return 100 end -- 100 examples
		for i=1,dataset:size() do 
		  local input = torch.randn(2);     -- normally distributed example in 2d
		  local output = torch.Tensor(1);
		  if input[1]*input[2]>0 then     -- calculate label for XOR function
		    output[1] = -1;
		  else
		    output[1] = 1
		  end
		  dataset[i] = {input, output}
		end
		*/
		Tensor[][] dataset = new Tensor[100][];
		for (int i = 0; i < dataset.length; ++ i) {
			Tensor input = randn(2);
			Tensor output = new Tensor(1);
			if (input.get(1) * input.get(2) > 0) {
				output.set(-1, 1);
			}
			else {
				output.set(1, 1);
			}
			dataset[i] = new Tensor[] { input, output };
		}

		/*
		require "nn"
		mlp = nn.Sequential();  -- make a multi-layer perceptron
		inputs = 2; outputs = 1; HUs = 20; -- parameters
		mlp:add(nn.Linear(inputs, HUs))
		mlp:add(nn.Tanh())
		mlp:add(nn.Linear(HUs, outputs))
		*/
		Container mlp = new Sequential();
		int inputs = 2, outputs = 1, HUs = 20;
		mlp.add(new Linear(inputs, HUs));
		mlp.add(new Tanh());
		mlp.add(new Linear(HUs, outputs));

		/*
		criterion = nn.MSECriterion()  
		trainer = nn.StochasticGradient(mlp, criterion)
		trainer.learningRate = 0.01
		trainer:train(dataset)
		*/

		Criterion criterion = new MSECriterion();
		StochasticGradient trainer = new StochasticGradient(mlp, criterion);
		trainer.learningRate = 0.01;
		trainer.train(dataset);

		/*
		x = torch.Tensor(2)
		x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
		x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
		x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
		x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
		*/
		Tensor x = new Tensor(2);
		x.set(0.5, 1); x.set(0.5, 2); System.out.println(mlp.forward(x));
		x.set(0.5, 1); x.set(-0.5, 2); System.out.println(mlp.forward(x));
		x.set(-0.5, 1); x.set(0.5, 2); System.out.println(mlp.forward(x));
		x.set(-0.5, 1); x.set(-0.5, 2); System.out.println(mlp.forward(x));
	}

	static void testStochasticGradient2() {
		/*
		require "nn"
		mlp = nn.Sequential();  -- make a multi-layer perceptron
		inputs = 2; outputs = 1; HUs = 20; -- parameters
		mlp:add(nn.Linear(inputs, HUs))
		mlp:add(nn.Tanh())
		mlp:add(nn.Linear(HUs, outputs))

		criterion = nn.MSECriterion()  
		*/
		Container mlp = new Sequential();
		int inputs = 2, outputs = 1, HUs = 20;
		mlp.add(new Linear(inputs, HUs));
		mlp.add(new Tanh());
		mlp.add(new Linear(HUs, outputs));

		Criterion criterion = new MSECriterion();

		/*
		for i = 1,2500 do
		  -- random sample
		  local input= torch.randn(2);     -- normally distributed example in 2d
		  local output= torch.Tensor(1);
		  if input[1]*input[2] > 0 then  -- calculate label for XOR function
		    output[1] = -1
		  else
		    output[1] = 1
		  end

		  -- feed it to the neural network and the criterion
		  criterion:forward(mlp:forward(input), output)

		  -- train over this example in 3 steps
		  -- (1) zero the accumulation of the gradients
		  mlp:zeroGradParameters()
		  -- (2) accumulate gradients
		  mlp:backward(input, criterion:backward(mlp.output, output))
		  -- (3) update parameters with a 0.01 learning rate
		  mlp:updateParameters(0.01)
		end
		*/
		for (int i = 0; i < 2500; ++ i) {
			Tensor input = randn(2);
			Tensor output = new Tensor(1);
			if (input.get(1) * input.get(2) > 0) {
				output.set(-1, 1);
			}
			else {
				output.set(1, 1);
			}

			criterion.forward(mlp.forward(input), output);
			mlp.zeroGradParameters();
			mlp.backward(input, criterion.backward(mlp.output(), output));
			mlp.updateParameters(0.01);
		}

		/*
		x = torch.Tensor(2)
		x[1] =  0.5; x[2] =  0.5; print(mlp:forward(x))
		x[1] =  0.5; x[2] = -0.5; print(mlp:forward(x))
		x[1] = -0.5; x[2] =  0.5; print(mlp:forward(x))
		x[1] = -0.5; x[2] = -0.5; print(mlp:forward(x))
		*/
		Tensor x = new Tensor(2);
		x.set(0.5, 1); x.set(0.5, 2); System.out.println(mlp.forward(x));
		x.set(0.5, 1); x.set(-0.5, 2); System.out.println(mlp.forward(x));
		x.set(-0.5, 1); x.set(0.5, 2); System.out.println(mlp.forward(x));
		x.set(-0.5, 1); x.set(-0.5, 2); System.out.println(mlp.forward(x));
	}

	public static void main(String ... args) {
		tic();

		//testJacobian1();
		//testJacobian2();
		//testJacobian3();
		//testJacobian4();
		//testStochasticGradient1();
		testStochasticGradient2();
		
		toc();
	}
	
}