package nn;

// local StochasticGradient = torch.class('nn.StochasticGradient')

public class StochasticGradient
{

// function StochasticGradient:__init(module, criterion)
//    self.learningRate = 0.01
//    self.learningRateDecay = 0
//    self.maxIteration = 25
//    self.shuffleIndices = true
//    self.module = module
//    self.criterion = criterion
//    self.verbose = true
// end

	public double learningRate;
	public double learningRateDecay;
	public int maxIteration;
	public boolean shuffleIndices;
	public Module module;
	public Criterion criterion;
	public boolean verbose;

	public StochasticGradient(Module module, Criterion criterion) {
		learningRate = 0.01;
		learningRateDecay = 0;
		maxIteration = 25;
		shuffleIndices = true;
		this.module = module;
		this.criterion = criterion;
		this.verbose = true;
	}

// function StochasticGradient:train(dataset)
//    local iteration = 1
//    local currentLearningRate = self.learningRate
//    local module = self.module
//    local criterion = self.criterion

//    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
//    if not self.shuffleIndices then
//       for t = 1,dataset:size() do
//          shuffledIndices[t] = t
//       end
//    end

//    print("# StochasticGradient: training")

//    while true do
//       local currentError = 0
//       for t = 1,dataset:size() do
//          local example = dataset[shuffledIndices[t]]
//          local input = example[1]
//          local target = example[2]

//          currentError = currentError + criterion:forward(module:forward(input), target)

//          module:updateGradInput(input, criterion:updateGradInput(module.output, target))
//          module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)

//          if self.hookExample then
//             self.hookExample(self, example)
//          end
//       end

//       currentError = currentError / dataset:size()

//       if self.hookIteration then
//          self.hookIteration(self, iteration, currentError)
//       end

//       if self.verbose then
//          print("# current error = " .. currentError)
//       end
//       iteration = iteration + 1
//       currentLearningRate = self.learningRate/(1+iteration*self.learningRateDecay)
//       if self.maxIteration > 0 and iteration > self.maxIteration then
//          print("# StochasticGradient: you have reached the maximum number of iterations")
//          print("# training error = " .. currentError)
//          break
//       end
//    end
// end

	public void train(Tensor[][] dataset) {
		int iteration = 1;
		double currentLearningRate = learningRate;

		int[] shuffledIndices = new int[dataset.length];
		for (int i = 0; i < shuffledIndices.length; ++ i) shuffledIndices[i] = i;
		if (this.shuffleIndices) randperm(shuffledIndices);

		System.out.println("# StochasticGradient: training");

		while (true) {
			double currentError = 0;
			for (int t = 1; t <= dataset.length; ++ t) {
				Tensor[] example = dataset[shuffledIndices[t - 1]];
				Tensor input = example[0];
				Tensor target = example[1];

				currentError += criterion.forward(module.forward(input), target);
				module.updateGradInput(input, criterion.updateGradInput(module.output(), target));
				module.accUpdateGradParameters(input, criterion.gradInput(), currentLearningRate);
			}
			currentError /= dataset.length;

			if (verbose) {
				System.out.println("# current error = " + currentError);
			}
			++ iteration;
			currentLearningRate = learningRate / (1 + iteration * learningRateDecay);
			if (maxIteration > 0 && iteration > maxIteration) {
				System.out.println("# StochasticGradient: you have reached the maximum number of iterations");
				System.out.println("# training error = " + currentError);
				break;
			}
		}
	}

	static void randperm(int[] indices) {
		for (int i = indices.length; i >= 2; -- i) {
			int rand = TensorMath.random.nextInt(i);
			if (rand == i - 1) continue;
			int temp = indices[i - 1];
			indices[i - 1] = indices[rand];
			indices[rand] = temp;
		}
	}


}
