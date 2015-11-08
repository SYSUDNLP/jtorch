
package nn;

import java.util.function.Consumer;

public final class Parallel
{
	public static Tensor mm(Tensor res, Tensor mat1, Tensor mat2) {
		if (mat1.dim() != 2 || mat1.dim() != 2) throw new IllegalArgumentException("Invalid arguments.");
		if (mat1.size(2) != mat2.size(1)) throw new IllegalArgumentException("Size mismatch.");
		if (res == null) res = new Tensor(mat1.size(1), mat2.size(2));
		else assert(res.isSize(mat1.size(1), mat2.size(2)));
		mat1 = mat1.contiguous();
		mat2 = mat2.t().contiguous();
		_mmt(res,
			mat1.storage(), mat1.storageOffset(),
			mat2.storage(), mat2.storageOffset(), 
			mat1.size(2));
		return res;
	}

	public static Tensor mm(Tensor mat1, Tensor mat2) {
		return mm(null, mat1, mat2);
	}

	private static void _mmt(Tensor res,
							double[] mat, int matOffset,
							double[] matt, int mattOffset, 
							int vecSize)
	{
		//for (int i = 0; i < res.size(1); ++ i) {
		new Parallel(res.size(1), i -> {
			for (int j = 0; j < res.size(2); ++ j) {
				double res1 = _dotvv(vecSize, mat, matOffset + i * vecSize, matt, mattOffset + j * vecSize);
				res.set(res1, i+1, j+1);
			}
		});
	}

	public static double _dotvv(int vecSize, double[] vec1, int offset1, double[] vec2, int offset2) {
		double sum = 0;
		for (int i = 0; i < vecSize; ++ i) {
			sum += vec1[offset1 + i] * vec2[offset2 + i];
		}
		return sum;
	}

// [res] torch.bmm([res,] batch1, batch2)

	public static Tensor bmm(Tensor res, Tensor batch1, Tensor batch2) {
		System.out.println(batch1.dim());
		System.out.println(batch2.dim());
		System.out.println(batch1.size(1));
		System.out.println(batch2.size(1));
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

	/////////////////////////////////

	private Consumer<Integer> task;
	
	private int taskCount;
	private int nextTask;
			
	public Parallel(int taskCount, java.util.function.Consumer<Integer> task) {
		this.taskCount = taskCount;
		this.task = task;
		nextTask = 0;
		start();
	}
	
	private synchronized int getNextTask() {
		if (nextTask == taskCount) return -1;
		return nextTask ++;
	}
	
	private void threadLoop() {
		while (true) {
			int taskNum = getNextTask();
			if (taskNum == -1) break;
			try {
				task.accept(taskNum);
			} catch (Exception ex) {
				ex.printStackTrace(System.err);
			}
		}
	}

	private void start() {
		int threadCount = Runtime.getRuntime().availableProcessors();
		if (threadCount <= 0) threadCount = 1;
		Thread[] threads = new Thread[threadCount];
		for (int i = 0; i < threadCount; ++ i) {
			threads[i] = new Thread(this::threadLoop);
			threads[i].start();
		}
		for (int i = 0; i < threadCount; ++ i) {
			try {
				threads[i].join();
			} catch (InterruptedException ex) {
			}
		}
	}
	

}
