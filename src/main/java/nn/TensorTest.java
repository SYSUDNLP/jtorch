
package nn;

import static nn.TensorMath.*; // To directly access all static functions in nn.TensorMath

public final class TensorTest
{
	/*
	t = torch.Tensor(3,4,5)
	for i = 1,t:dim() do
		print('t:size(i)=', t:size(i), 't:stride(i)=', t:stride(i), '\n')
	end
	*/
	static void testSizeAndStride() {
		//nn.Tensor t1 = new nn.Tensor(3,4,5);
		//t1 = t1.transpose(1,3).range(); // range fill all elements with serial 1, 2, 3, ... 
		Tensor t1 = reshape(range(1, 5*4*3), 5,4,3);
		print(t1);
		
		nn.Tensor t2 = new nn.Tensor(new int[] {3,4,5}, new int[] {-1,-1,0});
		printInfo(t2);
		
		//nn.Tensor t3 = new nn.Tensor(new int[] {5,4,3}, new int[] {-1,2,-1});
		//t3 = t3.transpose(1,3).range();
		Tensor t3 = zeros(3,4,5);
		range(t3, 1, 3*4*5, 1);
		t3.set(12.34, 3,4,5);
		t3.set(43.21, 3,2,1);
		print(t3);
		// t3:storage():size()
				
		print(diag(diag(ones(5, 5))));
		print(new nn.Tensor(new double[][][] {{{1,2,3}, {4,5,6}}, {{11,12,13}, {14,15,16}}}));
		print(diag(ones(5))); // eye(5)
	}

	/*	
	t = torch.Tensor(3,4,5)
	for i = 1,t:size(1) do
		for j = 1,t:size(2) do
			t[i][j] = i * 10 + j;
		end
	end
	print(t)
	*/
	static void testGetAndSet() {
		nn.Tensor t1 = new nn.Tensor(3);
		for (int i = 1; i <= t1.size(1); ++ i) {
			t1.set(i * 10, i);
		}
		print(t1);

		nn.Tensor t2 = new nn.Tensor(3,4);
		for (int i = 1; i <= t2.size(1); ++ i) {
			for (int j = 1; j <= t2.size(2); ++ j) {
				t2.set(i * 10 + j, i, j);
			}
		}
		print(t2);
	}

	/*
	 x = torch.Tensor(4,5)
	 i = 0
	 
	 x:apply(function()
	 i = i + 1
	 return i
	 end)
	 */
/*
	static void testApply() {
		nn.Tensor t = new nn.Tensor(4,5).copy(range(1, 4*5));
		print(t);
		print(t.max());
		print(t.min());
		print(t.sum());
		print(t.mean());
		print(t.std());
	}
	*/
	
	/*
	 x = torch.Tensor(3,3)
	 y = torch.Tensor(9)
	 z = torch.Tensor(3,3)
	 
	 x:fill(1)
	 y:fill(2)
	 z:fill(3)
	 
	 x:map2(y, z, function(xx, yy, zz) return xx+yy*zz end)
	print(x)
	 */
	static void testMap() {
		nn.Tensor x = new nn.Tensor(3,3);
		nn.Tensor y = new nn.Tensor(9);
		nn.Tensor z = new nn.Tensor(3,3);

		x.fill(1);
		y.fill(2);
		z.fill(3);
	 
		x.map2(y, z, (xx, yy, zz) -> (xx+yy*zz));
		print(x);

		x.map(y, (xx, yy) -> ((xx+yy) / 2));
		print(x);
		
		x.copy(z);
		print(x);
	}
	
	/*
	 x2 = torch.Tensor({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
	print(x2)
	print(x2:narrow(1, 2, 2))
	print(x2:narrow(2, 2, 2))
	print(x2:narrow(2, 3, 2):copy(x2:narrow(2, 2, 2)))
	 */
	static void testNarrow() {
		nn.Tensor x2 = new nn.Tensor(new double[][][] {{{1,2,3,4},{5,6,7,8},{9,10,11,12}},
			{{13,14,15,16},{17,18,19,20},{21,22,23,24}}});
		//nn.Tensor x2 = new nn.Tensor(new double[][] {{1,2,3,4},{5,6,7,8},{9,10,11,12}});
		printInfo(x2.select(1,1));
		printInfo(x2.select(1,2));
		print(x2);
		
		printInfo(x2.narrow(1, 2, 1));
		print(x2.narrow(2, 2, 2));
		
		x2.narrow(2, 1, 2).copy(x2.narrow(2, 2, 2));
		print(x2);
		x2.narrow(1, 2, 1).narrow(2, 2, 2).fill(100);
		print(x2);
		
		x2 = new nn.Tensor(2,3,4,5);
		double[] s = x2.storage();
		for (int i = 0; i < s.length; ++ i) s[i] = i + 1;
		print(x2);
	}
	
	/*
	 x2 = torch.Tensor({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
	 print(x2)
	 print(x2:sub(2,2,2,3))
	 x2:sub(2,2,2,3):fill(100)
	 print(x2)
	 */
	static void testSub() {
		nn.Tensor x2 = new nn.Tensor(new double[][] {{1,2,3,4},{5,6,7,8},{9,10,11,12}});
		print(x2);
		print(x2.sub(2,2,2,3));
		x2.sub(2,2,2,3).fill(100);
		print(x2);
		x2.sub(2,2,-1,-1).fill(1000);
		print(x2);
		x2.sub(-1,-1).fill(10000);
		print(x2);
	}

	/*
	 x2 = torch.Tensor({{1,2,3,4},{5,6,7,8},{9,10,11,12}})
	 print(x2)
	 print(x2:select(1,2))
	 print(x2:select(2,4))
	 */
	static void testSelect() {
		Tensor x2 =  new Tensor(new double[][] {{1,2,3,4},{5,6,7,8},{9,10,11,12}});
		print(x2);
		print(x2.select(1,2));
		print(x2.select(2,4));
	}

	/*
	 */
	static void testExpand() {
		Tensor x2 = rand(1,3,1);
		printInfo(x2);
		print(x2);
		x2 = x2.expand(2,3,4);
		printInfo(x2);
		print(x2);
	}

	/*
	 */
	static void testSqueeze() {
		Tensor x2 = rand(1,3,1,4);
		print(x2);
		print(x2.squeeze());
		print(x2.squeeze(1));
		print(x2.squeeze(3));
	}

	/*
	*/
	static void testView() {
		Tensor x2 = rand(2,3);
		print(x2);
		print(x2.narrow(1,1,1));
		printInfo(x2.narrow(1,1,1));
		print(x2.narrow(1,2,1));
		printInfo(x2.narrow(1,2,1));
		print(x2.narrow(1,2,1).view(3,1));
	}

	static void testPermute() {
		Tensor x = reshape(range(1, 2*3*4), 2,3,4);
		print(x);
		x.select(3,4).fill(100);
		print(x);
	}

	/*
	t = torch.Tensor(3,4);
	t:copy(torch.range(1,t:nElement()))
	t:unfold(2,3,1)
	t:unfold(2,2,2)
	t:unfold(1,2,1)
	*/
	static void testInfold() {
		Tensor t = reshape(range(1, 3*4), 3,4);
		print(t.unfold(2,3,1));
		print(t.unfold(2,2,2));
		print(t.unfold(1,2,1));
	}

	/*
	x = torch.randn(3,4,5)
	x:split(2,1)
	x:split(3,2)
	x:split(2,3)	
	x:chunk(2,1)
	x:chunk(2,2)
	x:chunk(2,3)
	*/
	static void testSplitAndChunk() {
		Tensor x = randn(3,4,5);
		print(x.split(2,1));
		print(x.split(3,2));
		print(x.split(2,3));
		print(x.chunk(2,1));
		print(x.chunk(2,2));
		print(x.chunk(2,3));
	}

/////////////////////////////////////////////////////

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
		printInfo(t);
	}
	
	static String tensorInfo(Tensor t) {
		StringBuffer buf = new StringBuffer();
		buf.append("[nn.Tensor of size");
		int[] size = t.size();
		for (int i = 1; i <= size.length; ++ i) buf.append((i == 1 ? " " : "x") + size[i - 1]);
		buf.append("]\n");
		/*
		buf.append("size");
		int[] size = t.size();
		for (int i = 1; i <= size.length; ++ i) buf.append("\t" + size[i - 1]);
		buf.append("\n");
		buf.append("stride");
		int[] stride = t.stride();
		for (int i = 1; i <= stride.length; ++ i) buf.append("\t" + stride[i - 1]);
		buf.append("\n");
		buf.append("offset");
		for (int i = 1; i <= storageOffset.length; ++ i) buf.append("\t" + storageOffset[i - 1]);
		buf.append("\n");
		if (dimPermutation != null) {
			buf.append("dimPermutation");
			for (int i = 1; i <= dimPermutation.length; ++ i) buf.append("\t" + dimPermutation[i - 1]);
			buf.append("\n");
		}
		buf.append("storage\t" + t.storage().length + "\n");
		*/
		return buf.toString();
	}

	static void printInfo(Tensor t) {
		System.out.println(tensorInfo(t));
	}

	static void print(Tensor[] ts) {
		System.out.println("Array of " + ts.length + " tensors:\n");
		for (Tensor t : ts) {
			printInfo(t);
		}
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


/////////////////////////////////////////////////////


	static void testMath() {

		// import nn.Tensor;
		// import static nn.TensorMath.*; // To directly access all static functions in nn.TensorMath

		Tensor x1 = eye(3); // torch.eye(3)
		Tensor x2 = zeros(3,3); // torch.zeros(3)
		Tensor x3 = ones(3,3); // torch.ones(3)
		Tensor x4 = rand(3,3); // torch.rand(3)
		Tensor x5 = randn(3,3); // torch.randn(3)

		// every function has two versions:
		x5 = add(ones(3,3), eye(3)); // returns a new storage
		add(x5, ones(3,3), eye(3)); // use the existing storage of x5

		print(
			add(ones(3,3), eye(3))
		);

		print(
			csub(ones(3,3), eye(3))
		);

		print(
			add(eye(3), 2)
		);

		print(
			csub(eye(3), 2)
		);

		// torch.mm(torch.ones(3,4), torch.reshape(torch.range(1,4*5),4,5))
		print(
			mm(ones(3,4), reshape(range(1,4*5),4,5)) // matrix-matrix multiplication
		);

		/* torch.bmm(
				torch.ones(3,4):view(1,3,4):expand(2,3,4), 
				torch.reshape(torch.range(1,4*5),4,5):view(1,4,5):expand(2,4,5)
			) */
		print(
			bmm(		// batch matrix-matrix multiplication
				ones(3,4).view(1,3,4).expand(2,3,4), 
				reshape(range(1,4*5),4,5).view(1,4,5).expand(2,4,5)
			)
		);

		print(
			mul(eye(3), 2)
		);

		print(
			div(eye(3), 2)
		);

		print(
			inverse(add(eye(5), ones(5,5)))
		);

		Tensor b1 = rand(5000,5000);
		Tensor b2 = rand(5000,5000);
		tic();
		mm(b1,b2);
		toc();
		Parallel.mm(b1,b2);
		toc();

	}

	public static void main(String ... args) {
		testMath();
		//testSizeAndStride();
		//testGetAndSet();
		//testApply();
		//testMap();
		//testNarrow();
		//testSub();
		//testSelect();
		//testExpand();
		//testSqueeze();
		//testView();
		//testPermute();
		//testInfold();
		//testSplitAndChunk();
	}
	
}