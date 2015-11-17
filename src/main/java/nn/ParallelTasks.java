
package nn;

public final class ParallelTasks
{

// [res] torch.mm([res,] mat1, mat2)

	public static Tensor mm(Tensor res, Tensor mat1, Tensor mat2) {
		if (mat1.dim() != 2 || mat1.dim() != 2) throw new IllegalArgumentException("Invalid arguments.");
		if (mat1.size(2) != mat2.size(1)) throw new IllegalArgumentException("Size mismatch.");
		if (res == null) res = new Tensor(mat1.size(1), mat2.size(2));
		else assert(res.isSize(mat1.size(1), mat2.size(2)));

		mat1 = mat1.contiguous();
		mat2 = mat2.t().contiguous();
		Tensor res2 = res.contiguous();
		_mmt(res.size(1), res.size(2),
			res2.storage(), res2.storageOffset(),
			mat1.storage(), mat1.storageOffset(),
			mat2.storage(), mat2.storageOffset(), 
			mat1.size(2));
		if (! res.isContiguous()) {
			res.copy(res2);
		}
		return res;
	}

	public static Tensor mm(Tensor mat1, Tensor mat2) {
		return mm(null, mat1, mat2);
	}

	private static void _mmt(int size1, int size2,
							double[] res, int resOffset,
							double[] mat, int matOffset,
							double[] matt, int mattOffset, 
							int vecSize)
	{
		//for (int i = 0; i < res.size(1); ++ i) {
		new ParallelTasks( size1,
			(from, to) -> {
				for (int i = from; i < to; ++ i) {
					for (int j = 0; j < size2; ++ j) {
						res[resOffset + i * size2 + j] = _dotvv(vecSize, mat, matOffset + i * vecSize, matt, mattOffset + j * vecSize);
					}
				}
			}
		);
	}

	private static double _dotvv(int vecSize, double[] vec1, int offset1, double[] vec2, int offset2) {
		double sum = 0;
		for (int i = 0; i < vecSize; ++ i) {
			sum += vec1[offset1 + i] * vec2[offset2 + i];
		}
		return sum;
	}

// [res] torch.bmm([res,] batch1, batch2)

	public static Tensor bmm(Tensor res, Tensor batch1, Tensor batch2) {
		if (batch1.dim() != 3 || batch2.dim() != 3 || batch1.size(1) != batch2.size(1)) 
			throw new IllegalArgumentException("Invalid batch size.");
		if (res == null) res = new Tensor(batch1.size(1), batch1.size(2), batch2.size(3));
		else assert(res.isSize(batch1.size(1), batch1.size(2), batch2.size(3)));
		for (int i = 1; i <= batch1.size(1); ++ i) {
			mm(res.select(1, i), batch1.select(1, i), batch2.select(1, i));
		}
		return res;
	}

	public static Tensor bmm(Tensor mat1, Tensor mat2) {
		return bmm(null, mat1, mat2);
	}

// [res] torch.mv([res,] mat, vec)

	public static Tensor mv(Tensor res, Tensor mat, Tensor vec) {
		if (mat.dim() != 2 || vec.dim() != 1) throw new IllegalArgumentException("Invalid arguments.");
		return mm(mat, vec.view(vec.size(1), 1)).view(mat.size(1));
	}

	public static Tensor mv(Tensor mat, Tensor vec) {
		return mv(null, mat, vec);
	}

	/////////////////////////////////
	
	@FunctionalInterface
	interface Consumer2
	{
		public void accept(int from, int to);
	}

	public ParallelTasks(int taskCount, Consumer2 task) {
		int threadCount = Runtime.getRuntime().availableProcessors();
		if (threadCount <= 0) threadCount = 1;
		TaskThread[] threads = new TaskThread[threadCount];

		int tasksPerThread = (int)Math.ceil(taskCount / (double)threadCount);
		//System.out.println("threadCount = " + threadCount + ", tasksPerThread = " + tasksPerThread);

		int nextTask = 0;
		for (int i = 0; i < threadCount; ++ i) {
			int from = nextTask;
			int to = (nextTask += tasksPerThread);
			if (to > taskCount) to = taskCount;
			threads[i] = new TaskThread(task, from, to);
			threads[i].start();
		}

		for (int i = 0; i < threadCount; ++ i) {
			try {
				threads[i].join();
			} catch (InterruptedException ex) {
			}
		}
	}

	static class TaskThread extends Thread
	{
		public int from;
		public int to;
		public Consumer2 task;

		public TaskThread(Consumer2 task, int from, int to) {
			this.task = task;
			this.from = from;
			this.to = to;
		}

		public void run() {
			try {
				task.accept(from, to);
			} catch (Exception ex) {
				ex.printStackTrace(System.err);
			}
		}
	}

}
