
package nn;

// Tensor
// The Tensor class is probably the most important class in Torch.
// Almost every package depends on this class.
// It is the class for handling numeric data.
// As with pretty much anything in Torch7, tensors are serializable.

// Multi-dimensional matrix

public final class Tensor // DoubleTensor
{

// A Tensor is a potentially multi-dimensional matrix. 
// The number of dimensions is unlimited that can be created using LongStorage with more dimensions.

// Example:

// --- creation of a 4D-tensor 4x5x6x2
// z = torch.Tensor(4,5,6,2)
// --- for more dimensions, (here a 6D tensor) one can do:
// s = torch.LongStorage(6)
// s[1] = 4; s[2] = 5; s[3] = 6; s[4] = 2; s[5] = 7; s[6] = 3;
// x = torch.Tensor(s)

    private int[] size;
    private double[] storage;
    private int[] storageOffset;
    private int[] stride;
    private int[] dimPermutation;

    public Tensor() {
        size = new int[0];
        storage = new double[0];
        storageOffset = new int[1];
        stride = new int[0];
        dimPermutation = null;
    }

    public Tensor(int... size) {
        this.size = _copyIntArray(size);
        stride = _defaultStride(size);
        storageOffset = new int[size.length + 1];
        storage = new double[nElement()];
        dimPermutation = null;
    }

    private static int[] _copyIntArray(int[] array) {
        if (array == null) return null;
        int[] array2 = new int[array.length];
        _copyIntArray(array2, array, 0, array.length, 0);
        return array2;
    }

    private static void _copyIntArray(int[] to, int[] from, int start, int end, int offset) {
        for (int i = start; i < end; ++i) {
            to[i + offset] = from[i];
        }
    }

    private static int[] _defaultStride(int[] size) {
        int[] stride = new int[size.length];
        stride[size.length - 1] = 1;
        for (int i = size.length - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * size[i + 1];
        }
        return stride;
    }

// The number of dimensions of a Tensor can be queried by nDimension() or dim(). Size of the i-th dimension is returned by size(i).
// A LongStorage containing all the dimensions can be returned by size().
// > x:nDimension()
// 6
// > x:size()
// 4
// 5
// 6
// 2
// 7
// 3
// [torch.LongStorage of size 6]
// Internal data representation

    public int[] size() {
        return _physicalDimension(dimPermutation, size);
    }

    public int size(int dim) {
        dim = _physicalDimension(dimPermutation, dim);
        return size[dim - 1];
    }

    public int nDimension() {
        return size.length;
    }

    public int dim() {
        return size.length;
    }

    public int[] stride() {
        return _physicalDimension(dimPermutation, stride);
    }

    public int stride(int dim) {
        dim = _physicalDimension(dimPermutation, dim);
        return stride[dim - 1];
    }

    public double[] storage() {
        return storage;
    }

    private static int _physicalDimension(int[] dimPermutation, int dim) {
        if (dimPermutation == null) return dim;
        return dimPermutation[dim - 1];
    }

    private static int[] _physicalDimension(int[] dimPermutation, int[] dim) {
        if (dimPermutation == null) return dim;
        int[] dim2 = new int[dim.length];
        for (int i = 0; i < dim.length; ++i) {
            dim2[i] = dim[dimPermutation[i] - 1];
        }
        return dim2;
    }

    private static int[] _permuteIndex(int[] dimPermutation, int[] indexes) {
        if (dimPermutation == null) return indexes;
        int[] indexes2 = new int[indexes.length];
        for (int i = 0; i < indexes.length; ++i) {
            indexes2[dimPermutation[i] - 1] = indexes[i];
        }
        return indexes2;
    }

// The actual data of a Tensor is contained into a Storage. It can be accessed using storage().
// While the memory of a Tensor has to be contained in this unique Storage, it might not be contiguous:
// the first position used in the Storage is given by storageOffset() (starting at 1).
// And the jump needed to go from one element to another element in the i-th dimension is given by stride(i).
// In other words, given a 3D tensor

// x = torch.Tensor(7,7,7)
// accessing the element (3,4,5) can be done by

// > x[3][4][5]
// or equivalently (but slowly!)

    public double get(int... indexes) {
        return storage[_getOffset(indexes)];
    }

    public Tensor set(double value, int... indexes) {
        storage[_getOffset(indexes)] = value;
        return this;
    }

    private int _getOffset(int[] indexes) {
        if (size.length != indexes.length) {
            throw new IllegalArgumentException("The dimension of the given index is " + indexes.length + ", but it should be " + size.length + ".");
        }
        indexes = _permuteIndex(dimPermutation, indexes);
        int index = 0;
        for (int i = 0; i < size.length; ++i) {
            if (indexes[i] <= 0 || indexes[i] > size[i]) {
                throw new IndexOutOfBoundsException();
            }
            index += storageOffset[i] + stride[i] * (indexes[i] - 1);
        }
        return index + storageOffset[size.length];
    }

    public String toString() {
        java.text.DecimalFormat decimal = new java.text.DecimalFormat("#.####");
        StringBuffer buf = new StringBuffer();
        if (dim() == 1) {
            for (int i = 1; i <= size(1); ++i) {
                buf.append(decimal.format(get(i)) + "\n");
            }
        } else if (dim() == 2) {
            for (int i = 1; i <= size(1); ++i) {
                for (int j = 1; j <= size(2); ++j) {
                    buf.append(decimal.format(get(i, j)) + "\t");
                }
                buf.append("\n");
            }
        } else {
            buf.append(_toString(new java.util.ArrayList<Integer>()));
        }
        return buf.toString();
    }

    private String _toString(java.util.ArrayList<Integer> indexes1) {
        StringBuffer buf = new StringBuffer();
        if (dim() == 2) {
            buf.append("(");
            for (Integer index : indexes1) buf.append(index + ",");
            buf.append(".,.) = \n");
            buf.append(toString());
        } else {
            for (int i = 1; i <= size(1); ++i) {
                java.util.ArrayList<Integer> indexes2 = new java.util.ArrayList<>(indexes1);
                indexes2.add(i);
                buf.append(select(1, i)._toString(indexes2) + "\n");
            }
        }
        return buf.toString();
    }

// > x:storage()[x:storageOffset()
// +(3-1)*x:stride(1)+(4-1)*x:stride(2)+(5-1)*x:stride(3)]
// One could say that a Tensor is a particular way of viewing a Storage: a Storage only represents a chunk of memory,
// while the Tensor interprets this chunk of memory as having dimensions:

// x = torch.Tensor(4,5)
// s = x:storage()
// for i=1,s:size() do -- fill up the Storage
// s[i] = i
// end
// > x -- s is interpreted by x as a 2D matrix
// 1   2   3   4   5
// 6   7   8   9  10
// 11  12  13  14  15
// 16  17  18  19  20
// [torch.DoubleTensor of dimension 4x5]
// Note also that in Torch7 elements in the same row [elements along the last dimension] are contiguous in memory for a matrix [tensor]:

// x = torch.Tensor(4,5)
// i = 0

// x:apply(function()
// i = i + 1
// return i
// end)

    public Tensor apply(java.util.function.Supplier<Double> supplier) {
        int nel = nElement();
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        for (int i = 0; i < nel; ++i) {
            int[] ix = iitx.next();
            set(supplier.get(), ix);
        }
        return this;
    }

    public Tensor apply(java.util.function.Function<Double, Double> function) {
        int nel = nElement();
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        for (int i = 0; i < nel; ++i) {
            int[] ix = iitx.next();
            double x1 = get(ix);
            set(function.apply(x1), ix);
        }
        return this;
    }

    public static void reduce(Tensor x, java.util.function.Consumer<Double> consumer) {
        int nel = x.nElement();
        TensorIndexesIterator iitx = new TensorIndexesIterator(x);
        for (int i = 0; i < nel; ++i) {
            consumer.accept(x.get(iitx.next()));
        }
    }

    @FunctionalInterface
    interface Consumer2<T1, T2> {
        public void accept(T1 t1, T2 t2);
    }

    public static void reduce(Tensor x, Tensor y, Consumer2<Double, Double> consumer2) {
        int nel = x.nElement();
        if (nel != y.nElement()) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(x);
        TensorIndexesIterator iity = new TensorIndexesIterator(y);
        for (int i = 0; i < nel; ++i) {
            consumer2.accept(x.get(iitx.next()), y.get(iity.next()));
        }
    }

// > x
// 1   2   3   4   5
// 6   7   8   9  10
// 11  12  13  14  15
// 16  17  18  19  20
// [torch.DoubleTensor of dimension 4x5]

// > x:stride()
// 5
// 1  -- element in the last dimension are contiguous!
// [torch.LongStorage of size 2]
// This is exactly like in C (and not Fortran).

// Tensors of different types

// Actually, several types of Tensor exists:

// ByteTensor -- contains unsigned chars
// CharTensor -- contains signed chars
// ShortTensor -- contains shorts
// IntTensor -- contains ints
// FloatTensor -- contains floats
// DoubleTensor -- contains doubles
// Most numeric operations are implemented only for FloatTensor and DoubleTensor. Other Tensor types are useful if you want to save memory space.

// Default Tensor type

// For convenience, an alias torch.Tensor is provided, which allows the user to write type-independent scripts,
// which can then ran after choosing the desired Tensor type with a call like

// torch.setdefaulttensortype('torch.FloatTensor')
// See torch.setdefaulttensortype for more details. By default, the alias "points" on torch.DoubleTensor.

// Efficient memory management

// All tensor operations in this class do not make any memory copy. All these methods transform the existing tensor,
// or return a new tensor referencing the same storage. This magical behavior is internally obtained by good usage of the stride() and storageOffset().
// Example:

// x = torch.Tensor(5):zero()
// > x
// 0
// 0
// 0
// 0
// 0
// [torch.DoubleTensor of dimension 5]
// > x:narrow(1, 2, 3):fill(1) -- narrow() returns a Tensor
// -- referencing the same Storage as x
// > x
// 0
// 1
// 1
// 1
// 0
// [torch.Tensor of dimension 5]
// If you really need to copy a Tensor, you can use the copy() method:

    public Tensor fill(double value) {
        return apply(() -> value);
    }

// y = torch.Tensor(x:size()):copy(x)
// Or the convenience method

    public Tensor copy(Tensor that) {
        return map(that, (x, y) -> y);
    }

// y = x:clone()

    public Tensor clone() {
        Tensor that = new Tensor(this.size());
        that.copy(this);
        return that;
    }

// We now describe all the methods for Tensor. If you want to specify the Tensor type, just replace Tensor by the name of the Tensor variant (like CharTensor).


// Tensor constructors

// Tensor constructors, create new Tensor object, optionally, allocating new memory.
// By default the elements of a newly allocated memory are not initialized, therefore, might contain arbitrary numbers.
// Here are several ways to construct a new Tensor.


// torch.Tensor()

// Returns an empty tensor.


// torch.Tensor(tensor)

// Returns a new tensor which reference the same Storage than the given tensor. The size, stride, and storage offset are the same than the given tensor.

// The new Tensor is now going to "view" the same storage as the given tensor.
// As a result, any modification in the elements of the Tensor will have a impact on the elements of the given tensor, and vice-versa. No memory copy!

    public Tensor(Tensor t) {
        set(t);
    }

// x = torch.Tensor(2,5):fill(3.14)
// > x
// 3.1400  3.1400  3.1400  3.1400  3.1400
// 3.1400  3.1400  3.1400  3.1400  3.1400
// [torch.DoubleTensor of dimension 2x5]

// y = torch.Tensor(x)
// > y
// 3.1400  3.1400  3.1400  3.1400  3.1400
// 3.1400  3.1400  3.1400  3.1400  3.1400
// [torch.DoubleTensor of dimension 2x5]

// y:zero()
// > x -- elements of x are the same as y!
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 2x5]

// torch.Tensor(sz1 [,sz2 [,sz3 [,sz4]]]])

// Create a tensor up to 4 dimensions. The tensor size will be sz1 x sz2 x sx3 x sz4.


    // torch.Tensor(sizes, [strides])
    public Tensor(int[] size, int[] stride) {
        // negative strides : mean to choose the right strides
        this.size = _copyIntArray(size);
        this.stride = _copyIntArray(stride);
        storage = new double[_storageSizeAndStride(size, this.stride)];
        storageOffset = new int[size.length + 1];
        dimPermutation = null;
    }

    static int _storageSizeAndStride(int[] size, int[] stride) {
        int[] storageSize = new int[size.length + 1];
        storageSize[size.length] = 1;
        for (int i = size.length - 1; i >= 0; --i) {
            if (stride[i] < 0) {
                if (i == size.length - 1) {
                    stride[i] = 1;
                } else {
                    stride[i] = size[i + 1] * stride[i + 1];
                }
            }
            storageSize[i] = (size[i] - 1) * stride[i] + storageSize[i + 1];
        }
        return storageSize[0];
    }

// Create a tensor of any number of dimensions. The LongStorage sizes gives the size in each dimension of the tensor.
// The optional LongStorage strides gives the jump necessary to go from one element to the next one in the each dimension.
// Of course, sizes and strides must have the same number of elements.
// If not given, or if some elements of strides are negative, the stride() will be computed such that the tensor is as contiguous as possible in memory.

// Example, create a 4D 4x4x3x2 tensor:

// x = torch.Tensor(torch.LongStorage({4,4,3,2}))
// Playing with the strides can give some interesting things:

// x = torch.Tensor(torch.LongStorage({4}), torch.LongStorage({0})):zero() -- zeroes the tensor
// x[1] = 1 -- all elements point to the same address!
// > x
// 1
// 1
// 1
// 1
// [torch.DoubleTensor of dimension 4]
// Note that negative strides are not allowed, and, if given as argument when constructing the Tensor,
// will be interpreted as //choose the right stride such that the Tensor is contiguous in memory//.

// Note this method cannot be used to create torch.LongTensors. The constructor from a storage will be used:

// a = torch.LongStorage({1,2}) -- We have a torch.LongStorage containing the values 1 and 2
// -- General case for TYPE ~= Long, e.g. for TYPE = Float:
// b = torch.FloatTensor(a)
// -- Creates a new torch.FloatTensor with 2 dimensions, the first of size 1 and the second of size 2
// > b:size()
// 1
// 2
// [torch.LongStorage of size 2]

// -- Special case of torch.LongTensor
// c = torch.LongTensor(a)
// -- Creates a new torch.LongTensor that uses a as storage and thus contains the values 1 and 2
// > c
// 1
// 2
// [torch.LongTensor of size 2]

// torch.Tensor(storage, [storageOffset, sizes, [strides]])

// Returns a tensor which uses the existing Storage storage, starting at position storageOffset (>=1).
// The size of each dimension of the tensor is given by the LongStorage sizes.

// If only storage is provided, it will create a 1D Tensor viewing the all Storage.

// The jump necessary to go from one element to the next one in each dimension is given by the optional argument LongStorage strides.
// If not given, or if some elements of strides are negative, the stride() will be computed such that the tensor is as contiguous as possible in memory.

// Any modification in the elements of the Storage will have an impact on the elements of the new Tensor, and vice-versa. There is no memory copy!

// -- creates a storage with 10 elements
// s = torch.Storage(10):fill(1)

// -- we want to see it as a 2x5 tensor
// x = torch.Tensor(s, 1, torch.LongStorage{2,5})
// > x
// 1  1  1  1  1
// 1  1  1  1  1
// [torch.DoubleTensor of dimension 2x5]

// x:zero()
// > s -- the storage contents have been modified
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// [torch.DoubleStorage of size 10]

// torch.Tensor(storage, [storageOffset, sz1 [, st1 ... [, sz4 [, st4]]]])

// Convenience constructor (for the previous constructor) assuming a number of dimensions inferior or equal to 4.
// szi is the size in the i-th dimension, and sti is the stride in the i-th dimension.


// torch.Tensor(table)

// The argument is assumed to be a Lua array of numbers. The constructor returns a new Tensor of the size of the table, containing all the table elements.
// The table might be multi-dimensional.

// Example:

// > torch.Tensor({{1,2,3,4}, {5,6,7,8}})
// 1  2  3  4
// 5  6  7  8
// [torch.DoubleTensor of dimension 2x4]
// A note on function calls

    public Tensor copy(double[] y) {
        int nel = nElement();
        if (nel != y.length) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        for (int i = 0; i < y.length; ++i) {
            int[] ix = iitx.next();
            double x1 = get(ix);
            set(y[i], ix);
        }
        return this;
    }

    public Tensor copy(double[][] y) {
        int nel = nElement();
        if (nel != y.length * y[0].length) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        for (int i = 0; i < y.length; ++i) {
            for (int j = 0; j < y[0].length; ++j) {
                int[] ix = iitx.next();
                double x1 = get(ix);
                set(y[i][j], ix);
            }
        }
        return this;
    }

    public Tensor copy(double[][][] y) {
        int nel = nElement();
        if (nel != y.length * y[0].length * y[0][0].length) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        for (int i = 0; i < y.length; ++i) {
            for (int j = 0; j < y[0].length; ++j) {
                for (int k = 0; k < y[0][0].length; ++k) {
                    int[] ix = iitx.next();
                    double x1 = get(ix);
                    set(y[i][j][k], ix);
                }
            }
        }
        return this;
    }

    public Tensor(double[] values) {
        this(values.length);
        copy(values);
    }

    public Tensor(double[][] values) {
        this(values.length, values[0].length);
        copy(values);
    }

    public Tensor(double[][][] values) {
        this(values.length, values[0].length, values[0][0].length);
        copy(values);
    }

// The rest of this guide will present many functions that can be used to manipulate tensors.
// Most functions have been defined so that they can be called flexibly, either in an object-oriented "method call" style
// i.e. src:function(...) or a more "functional" style torch.function(src, ...), where src is a tensor.
// Note that these different invocations may differ in whether they modify the tensor in-place, or create a new tensor.
// Additionally, some functions can be called in the form dst:function(src, ...)
// which usually suggests that the result of the operation on the src tensor will be stored in the tensor dst.
// Further details are given in the individual function definitions,
// below, but it should be noted that the documentation is currently incomplete in this regard, and readers are encouraged to experiment in an interactive session.

// Cloning


// [Tensor] clone()

// Returns a clone of a tensor. The memory is copied.

// i = 0
// x = torch.Tensor(5):apply(function(x)
// i = i + 1
// return i
// end)
// > x
// 1
// 2
// 3
// 4
// 5
// [torch.DoubleTensor of dimension 5]

// -- create a clone of x
// y = x:clone()
// > y
// 1
// 2
// 3
// 4
// 5
// [torch.DoubleTensor of dimension 5]

// -- fill up y with 1
// y:fill(1)
// > y
// 1
// 1
// 1
// 1
// 1
// [torch.DoubleTensor of dimension 5]

// -- the contents of x were not changed:
// > x
// 1
// 2
// 3
// 4
// 5
// [torch.DoubleTensor of dimension 5]

// [Tensor] contiguous

// If the given Tensor contents are contiguous in memory, returns the exact same Tensor (no memory copy).
// Otherwise (not contiguous in memory), returns a clone (memory copy).
// x = torch.Tensor(2,3):fill(1)
// > x
// 1  1  1
// 1  1  1
// [torch.DoubleTensor of dimension 2x3]

// -- x is contiguous, so y points to the same thing
// y = x:contiguous():fill(2)
// > y
// 2  2  2
// 2  2  2
// [torch.DoubleTensor of dimension 2x3]

// -- contents of x have been changed
// > x
// 2  2  2
// 2  2  2
// [torch.DoubleTensor of dimension 2x3]

// -- x:t() is not contiguous, so z is a clone
// z = x:t():contiguous():fill(3.14)
// > z
// 3.1400  3.1400
// 3.1400  3.1400
// 3.1400  3.1400
// [torch.DoubleTensor of dimension 3x2]

// -- contents of x have not been changed
// > x
// 2  2  2
// 2  2  2
// [torch.DoubleTensor of dimension 2x3]

    static boolean _compareIntArrays(int[] array1, int[] array2) {
        if (array1.length != array2.length) return false;
        for (int i = 0; i < array1.length; ++i) {
            if (array1[i] != array2[i]) return false;
        }
        return true;
    }

    public boolean isContiguous() {
        return _compareIntArrays(stride, _defaultStride(size)) && dimPermutation == null;
    }

    public Tensor contiguous() {
        if (isContiguous()) return this;
        return this.clone();
    }

// [Tensor or string] type(type)

// If type is nil, returns a string containing the type name of the given tensor.

// = torch.Tensor():type()
// torch.DoubleTensor
// If type is a string describing a Tensor type, and is equal to the given tensor typename, returns the exact same tensor (//no memory copy//).

// x = torch.Tensor(3):fill(3.14)
// > x
// 3.1400
// 3.1400
// 3.1400
// [torch.DoubleTensor of dimension 3]

// y = x:type('torch.DoubleTensor')
// > y
// 3.1400
// 3.1400
// 3.1400
// [torch.DoubleTensor of dimension 3]

// -- zero y contents
// y:zero()

// -- contents of x have been changed
// > x
// 0
// 0
// 0
// [torch.DoubleTensor of dimension 3]
// If type is a string describing a Tensor type, different from the type name of the given Tensor, returns a new Tensor of the specified type,
// whose contents corresponds to the contents of the original Tensor, casted to the given type (//memory copy occurs, with possible loss of precision//).

// x = torch.Tensor(3):fill(3.14)
// > x
// 3.1400
// 3.1400
// 3.1400
// [torch.DoubleTensor of dimension 3]

// y = x:type('torch.IntTensor')
// > y
// 3
// 3
// 3
// [torch.IntTensor of dimension 3]

// [Tensor] typeAs(tensor)

// Convenience method for the type method. Equivalent to

// type(tensor:type())

// [boolean] isTensor(object)

// Returns true iff the provided object is one of the torch.*Tensor types.

// > torch.isTensor(torch.randn(3,4))
// true

// > torch.isTensor(torch.randn(3,4)[1])
// true

// > torch.isTensor(torch.randn(3,4)[1][2])
// false

// [Tensor] byte(), char(), short(), int(), long(), float(), double()


// Convenience methods for the type method. For e.g.,

// x = torch.Tensor(3):fill(3.14)
// > x
// 3.1400
// 3.1400
// 3.1400
// [torch.DoubleTensor of dimension 3]

// -- calling type('torch.IntTensor')
// > x:type('torch.IntTensor')
// 3
// 3
// 3
// [torch.IntTensor of dimension 3]


// -- is equivalent to calling int()
// > x:int()
// 3
// 3
// 3
// [torch.IntTensor of dimension 3]
// Querying the size and structure


// [number] nDimension()

// Returns the number of dimensions in a Tensor.

// x = torch.Tensor(4,5) -- a matrix
// > x:nDimension()
// 2

// [number] dim()

// Same as nDimension().


// [number] size(dim)

// Returns the size of the specified dimension dim. Example:

// x = torch.Tensor(4,5):zero()
// > x
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 4x5]

// > x:size(2) -- gets the number of columns
// 5

// [LongStorage] size()

// Returns a LongStorage containing the size of each dimension of the tensor.

// x = torch.Tensor(4,5):zero()
// > x
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 4x5]

// > x:size()
// 4
// 5
// [torch.LongStorage of size 2]

// [LongStorage] #self

// Same as size() method.


// [number] stride(dim)

// Returns the jump necessary to go from one element to the next one in the specified dimension dim. Example:

// x = torch.Tensor(4,5):zero()
// > x
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 4x5]

// -- elements in a row are contiguous in memory
// > x:stride(2)
// 1

// -- to go from one element to the next one in a column
// -- we need here to jump the size of the row
// > x:stride(1)
// 5
// Note also that in Torch elements in the same row [elements along the last dimension] are contiguous in memory for a matrix [tensor].


// [LongStorage] stride()

// Returns the jump necessary to go from one element to the next one in each dimension. Example:

// x = torch.Tensor(4,5):zero()
// > x
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 4x5]

// > x:stride()
// 5
// 1 -- elements are contiguous in a row [last dimension]
// [torch.LongStorage of size 2]
// Note also that in Torch elements in the same row [elements along the last dimension] are contiguous in memory for a matrix [tensor].


// [Storage] storage()

// Returns the Storage used to store all the elements of the Tensor. Basically, a Tensor is a particular way of viewing a Storage.

// x = torch.Tensor(4,5)
// s = x:storage()
// for i=1,s:size() do -- fill up the Storage
// s[i] = i
// end

// > x -- s is interpreted by x as a 2D matrix
// 1   2   3   4   5
// 6   7   8   9  10
// 11  12  13  14  15
// 16  17  18  19  20
// [torch.DoubleTensor of dimension 4x5]

// [boolean] isContiguous()

// Returns true iff the elements of the Tensor are contiguous in memory.

// -- normal tensors are contiguous in memory
// x = torch.randn(4,5)
// > x:isContiguous()
// true

// -- y now "views" the 3rd column of x
// -- the storage of y is the same than x
// -- so the memory cannot be contiguous
// y = x:select(2, 3)
// > y:isContiguous()
// false

// -- indeed, to jump to one element to
// -- the next one, the stride is 5
// > y:stride()
// 5
// [torch.LongStorage of size 1]

// [boolean] isSize(storage)

// Returns true iff the dimensions of the Tensor match the elements of the storage.

// x = torch.Tensor(4,5)
// y = torch.LongStorage({4,5})
// z = torch.LongStorage({5,4,1})
// > x:isSize(y)
// true

// > x:isSize(z)
// false

// > x:isSize(x:size())
// true

    public boolean isSize(int... size) {
        return _compareIntArrays(size(), size);
    }

// [boolean] isSameSizeAs(tensor)

// Returns true iff the dimensions of the Tensor and the argument Tensor are exactly the same.

// x = torch.Tensor(4,5)
// y = torch.Tensor(4,5)
// > x:isSameSizeAs(y)
// true

// y = torch.Tensor(4,6)
// > x:isSameSizeAs(y)
// false

    public boolean isSameSizeAs(Tensor that) {
        return _compareIntArrays(size(), that.size());
    }

// [number] nElement()

// Returns the number of elements of a tensor.

// x = torch.Tensor(4,5)
// > x:nElement() -- 4x5 = 20!
// 20

    public int nElement() {
        return _totalSize(size);
    }

    static int _totalSize(int[] size) {
        int elements = size[0];
        for (int i = 1; i < size.length; ++i) {
            elements *= size[i];
        }
        return elements;
    }

// [number] storageOffset()

// Return the first index (starting at 1) used in the tensor's storage.

    public int storageOffset() {
        return _sumIntArray(storageOffset);
    }

    private static int _sumIntArray(int[] array) {
        int sum = 0;
        for (int i = 0; i < array.length; ++i) {
            sum += array[i];
        }
        return sum;
    }

// Querying elements

// Elements of a tensor can be retrieved with the [index] operator.

// If index is a number, [index] operator is equivalent to a select(1, index).
// If the tensor has more than one dimension, this operation returns a slice of the tensor that shares the same underlying storage.
// If the tensor is a 1D tensor, it returns the value at index in this tensor.

// If index is a table, the table must contain n numbers, where n is the number of dimensions of the Tensor. It will return the element at the given position.

// In the same spirit, index might be a LongStorage, specifying the position (in the Tensor) of the element to be retrieved.

// If index is a ByteTensor in which each element is 0 or 1 then it acts as a selection mask used to extract a subset of the original tensor.
// This is particularly useful with logical operators like torch.le.

// Example:

// x = torch.Tensor(3,3)
// i = 0; x:apply(function() i = i + 1; return i end)
// > x
// 1  2  3
// 4  5  6
// 7  8  9
// [torch.DoubleTensor of dimension 3x3]

// > x[2] -- returns row 2
// 4
// 5
// 6
// [torch.DoubleTensor of dimension 3]

// > x[2][3] -- returns row 2, column 3
// 6

// > x[{2,3}] -- another way to return row 2, column 3
// 6

// > x[torch.LongStorage{2,3}] -- yet another way to return row 2, column 3
// 6

// > x[torch.le(x,3)] -- torch.le returns a ByteTensor that acts as a mask
// 1
// 2
// 3
// [torch.DoubleTensor of dimension 3]

    public Tensor get(int[]... ranges) {
        if (ranges.length != dim()) {
            throw new IllegalArgumentException("Wrong number of dimensions.");
        }
        Tensor t = this;
        for (int i = 1; i <= ranges.length; ++i) {
            int from = 1;
            int to = size(i);
            if (ranges[i - 1].length == 1) {
                from = (ranges[i - 1][0] < 0 ? size(i) : ranges[i - 1][0]);
                to = from;
            }
            if (ranges[i - 1].length == 2) {
                to = (ranges[i - 1][1] < 0 ? size(i) : ranges[i - 1][1]);
            }
            t = t.narrow(i, from, to - from + 1);
        }
        return t;
    }

// Referencing a tensor to an existing tensor or chunk of memory

// A Tensor being a way of viewing a Storage, it is possible to "set" a Tensor such that it views an existing Storage.

// Note that if you want to perform a set on an empty Tensor like

// y = torch.Storage(10)
// x = torch.Tensor()
// x:set(y, 1, 10)
// you might want in that case to use one of the equivalent constructor.

// y = torch.Storage(10)
// x = torch.Tensor(y, 1, 10)

// [self] set(tensor)

// The Tensor is now going to "view" the same storage as the given tensor.
// As the result, any modification in the elements of the Tensor will have an impact on the elements of the given tensor, and vice-versa.
// This is an efficient method, as there is no memory copy!

// x = torch.Tensor(2,5):fill(3.14)
// > x
// 3.1400  3.1400  3.1400  3.1400  3.1400
// 3.1400  3.1400  3.1400  3.1400  3.1400
// [torch.DoubleTensor of dimension 2x5]

// y = torch.Tensor():set(x)
// > y
// 3.1400  3.1400  3.1400  3.1400  3.1400
// 3.1400  3.1400  3.1400  3.1400  3.1400
// [torch.DoubleTensor of dimension 2x5]

// y:zero()
// > x -- elements of x are the same than y!
// 0 0 0 0 0
// 0 0 0 0 0
// [torch.DoubleTensor of dimension 2x5]

    public Tensor set(Tensor that) {
        storage = that.storage;
        storageOffset = _copyIntArray(that.storageOffset);
        size = _copyIntArray(that.size);
        stride = _copyIntArray(that.stride);
        dimPermutation = _copyIntArray(that.dimPermutation);
        return this;
    }

// [self] set(storage, [storageOffset, sizes, [strides]])

// The Tensor is now going to "view" the given storage, starting at position storageOffset (>=1) with the given dimension sizes and the optional given strides.
// As the result, any modification in the elements of the Storage will have a impact on the elements of the Tensor, and vice-versa.
// This is an efficient method, as there is no memory copy!

// If only storage is provided, the whole storage will be viewed as a 1D Tensor.

// -- creates a storage with 10 elements
// s = torch.Storage(10):fill(1)

// -- we want to see it as a 2x5 tensor
// sz = torch.LongStorage({2,5})
// x = torch.Tensor()
// x:set(s, 1, sz)
// > x
// 1  1  1  1  1
// 1  1  1  1  1
// [torch.DoubleTensor of dimension 2x5]

// x:zero()
// > s -- the storage contents have been modified
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// 0
// [torch.DoubleStorage of size 10]

// [self] set(storage, [storageOffset, sz1 [, st1 ... [, sz4 [, st4]]]])

// This is a "shortcut" for previous method. It works up to 4 dimensions. szi is the size of the i-th dimension of the tensor. sti is the stride in the i-th dimension.

// Copying and initializing


// [self] copy(tensor)

// Replace the elements of the Tensor by copying the elements of the given tensor. The number of elements must match, but the sizes might be different.

// x = torch.Tensor(4):fill(1)
// y = torch.Tensor(2,2):copy(x)
// > x
// 1
// 1
// 1
// 1
// [torch.DoubleTensor of dimension 4]

// > y
// 1  1
// 1  1
// [torch.DoubleTensor of dimension 2x2]
// If a different type of tensor is given, then a type conversion occurs, which, of course, might result in loss of precision.


// [self] fill(value)

// Fill the tensor with the given value.

// > torch.DoubleTensor(4):fill(3.14)
// 3.1400
// 3.1400
// 3.1400
// 3.1400
// [torch.DoubleTensor of dimension 4]

// [self] zero()

// Fill the tensor with zeros.

// > torch.Tensor(4):zero()
// 0
// 0
// 0
// 0
// [torch.DoubleTensor of dimension 4]

// Resizing

// When resizing to a larger size, the underlying Storage is resized to fit all the elements of the Tensor.

// When resizing to a smaller size, the underlying Storage is not resized.

// Important note: the content of a Tensor after resizing is undetermined as strides might have been completely changed.
// In particular, the elements of the resized tensor are contiguous in memory.


// [self] resizeAs(tensor)

// Resize the tensor as the given tensor (of the same type).


// [self] resize(sizes)

// Resize the tensor according to the given LongStorage sizes.


// [self] resize(sz1 [,sz2 [,sz3 [,sz4]]]])

// Convenience method of the previous method, working for a number of dimensions up to 4.

// Extracting sub-tensors

// Each of these methods returns a Tensor which is a sub-tensor of the given tensor, with the same Storage.
// Hence, any modification in the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.

// These methods are very fast, as they do not involve any memory copy.


// [self] narrow(dim, index, size)

    public Tensor narrow(int dim, int index, int size) {
        dim = _physicalDimension(dimPermutation, dim);
        Tensor t = new Tensor(this);
        if (index <= 0 || index > t.size[dim - 1]) throw new IndexOutOfBoundsException();
        if (size <= 0) throw new IllegalArgumentException("'size' must be positive.");
        if (index + size - 1 > t.size[dim - 1]) throw new IndexOutOfBoundsException();
        t.storageOffset[dim - 1] += (index - 1) * t.stride[dim - 1];
        t.size[dim - 1] = size;
        return t;
    }

// Returns a new Tensor which is a narrowed version of the current one: the dimension dim is narrowed from index to index+size-1.

// x = torch.Tensor(5, 6):zero()
// > x

// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// [torch.DoubleTensor of dimension 5x6]

// y = x:narrow(1, 2, 3) -- narrow dimension 1 from index 2 to index 2+3-1
// y:fill(1) -- fill with 1
// > y
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// [torch.DoubleTensor of dimension 3x6]

// > x -- memory in x has been modified!
// 0  0  0  0  0  0
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// [Tensor] sub(dim1s, dim1e ... [, dim4s [, dim4e]])

// This method is equivalent to do a series of narrow up to the first 4 dimensions.
// It returns a new Tensor which is a sub-tensor going from index dimis to dimie in the i-th dimension.
// Negative values are interpreted index starting from the end: -1 is the last index, -2 is the index before the last index, ...

// x = torch.Tensor(5, 6):zero()
// > x
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// [torch.DoubleTensor of dimension 5x6]

// y = x:sub(2,4):fill(1) -- y is sub-tensor of x:
// > y                    -- dimension 1 starts at index 2, ends at index 4
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// [torch.DoubleTensor of dimension 3x6]

// > x                    -- x has been modified!
// 0  0  0  0  0  0
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 1  1  1  1  1  1
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// z = x:sub(2,4,3,4):fill(2) -- we now take a new sub-tensor
// > z                        -- dimension 1 starts at index 2, ends at index 4
// -- dimension 2 starts at index 3, ends at index 4
// 2  2
// 2  2
// 2  2
// [torch.DoubleTensor of dimension 3x2]

// > x                        -- x has been modified
// 0  0  0  0  0  0
// 1  1  2  2  1  1
// 1  1  2  2  1  1
// 1  1  2  2  1  1
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// > y                        -- y has been modified
// 1  1  2  2  1  1
// 1  1  2  2  1  1
// 1  1  2  2  1  1
// [torch.DoubleTensor of dimension 3x6]

// > y:sub(-1, -1, 3, 4)      -- negative values = bounds
// 2  2
// [torch.DoubleTensor of dimension 1x2]

    public Tensor sub(int... ranges) {
        if (ranges.length % 2 != 0) {
            throw new IllegalArgumentException("# of arguments should be even.");
        }
        if (ranges.length / 2 > dim()) {
            throw new IllegalArgumentException("Too many arguments.");
        }
        Tensor t = this;
        for (int i = 1; i <= ranges.length / 2; ++i) {
            if (ranges[i * 2 - 2] < 0) ranges[i * 2 - 2] = size(i);
            if (ranges[i * 2 - 1] < 0) ranges[i * 2 - 1] = size(i);
            t = t.narrow(i, ranges[i * 2 - 2], ranges[i * 2 - 1] - ranges[i * 2 - 2] + 1);
        }
        return t;
    }

// [Tensor] select(dim, index)

// Returns a new Tensor which is a tensor slice at the given index in the dimension dim.
// The returned tensor has one less dimension: the dimension dim is removed. As a result, it is not possible to select() on a 1D tensor.

// Note that "selecting" on the first dimension is equivalent to use the [] operator

// x = torch.Tensor(5,6):zero()
// > x
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// [torch.DoubleTensor of dimension 5x6]

// y = x:select(1, 2):fill(2) -- select row 2 and fill up
// > y
// 2
// 2
// 2
// 2
// 2
// 2
// [torch.DoubleTensor of dimension 6]

// > x
// 0  0  0  0  0  0
// 2  2  2  2  2  2
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// z = x:select(2,5):fill(5) -- select column 5 and fill up
// > z
// 5
// 5
// 5
// 5
// 5
// [torch.DoubleTensor of dimension 5]

// > x
// 0  0  0  0  5  0
// 2  2  2  2  5  2
// 0  0  0  0  5  0
// 0  0  0  0  5  0
// 0  0  0  0  5  0
// [torch.DoubleTensor of dimension 5x6]

    public Tensor select(int dim, int index) {
        Tensor t = narrow(dim, index, 1);

        int dim0 = dim;
        dim = _physicalDimension(dimPermutation, dim);
        t.storageOffset[dim] += t.storageOffset[dim - 1];

        t.storageOffset = _deleteIntArray(t.storageOffset, dim);
        t.size = _deleteIntArray(t.size, dim);
        t.stride = _deleteIntArray(t.stride, dim);
        t.dimPermutation = _deletePermutation(t.dimPermutation, dim0);
        return t;
    }

    static int[] _deleteIntArray(int[] array, int dim) {
        int[] array2 = new int[array.length - 1];
        _copyIntArray(array2, array, 0, dim - 1, 0);
        _copyIntArray(array2, array, dim, array.length, -1);
        return array2;
    }

    static int[] _deletePermutation(int[] dimPermutation, int dim0) {
        if (dimPermutation == null) return null;
        int dim = dimPermutation[dim0 - 1];
        dimPermutation = _deleteIntArray(dimPermutation, dim0);
        for (int i = 0; i < dimPermutation.length; ++i) {
            if (dimPermutation[i] > dim) --dimPermutation[i];
        }
        return _validateDimPermutation(dimPermutation);
    }

// [Tensor] [{ dim1,dim2,... }] or [{ {dim1s,dim1e}, {dim2s,dim2e} }]

// The indexing operator [] can be used to combine narrow/sub and select in a concise and efficient way. It can also be used to copy, and fill (sub) tensors.

// This operator also works with an input mask made of a ByteTensor with 0 and 1 elements, e.g with a logical operator.

// x = torch.Tensor(5, 6):zero()
// > x
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// 0 0 0 0 0 0
// [torch.DoubleTensor of dimension 5x6]

// x[{ 1,3 }] = 1 -- sets element at (i=1,j=3) to 1
// > x
// 0  0  1  0  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// x[{ 2,{2,4} }] = 2  -- sets a slice of 3 elements to 2
// > x
// 0  0  1  0  0  0
// 0  2  2  2  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// 0  0  0  0  0  0
// [torch.DoubleTensor of dimension 5x6]

// x[{ {},4 }] = -1 -- sets the full 4th column to -1
// > x
// 0  0  1 -1  0  0
// 0  2  2 -1  0  0
// 0  0  0 -1  0  0
// 0  0  0 -1  0  0
// 0  0  0 -1  0  0
// [torch.DoubleTensor of dimension 5x6]

// x[{ {},2 }] = torch.range(1,5) -- copy a 1D tensor to a slice of x
// > x

// 0  1  1 -1  0  0
// 0  2  2 -1  0  0
// 0  3  0 -1  0  0
// 0  4  0 -1  0  0
// 0  5  0 -1  0  0
// [torch.DoubleTensor of dimension 5x6]

// x[torch.lt(x,0)] = -2 -- sets all negative elements to -2 via a mask
// > x

// 0  1  1 -2  0  0
// 0  2  2 -2  0  0
// 0  3  0 -2  0  0
// 0  4  0 -2  0  0
// 0  5  0 -2  0  0
// [torch.DoubleTensor of dimension 5x6]

// [Tensor] index(dim, index)

// Returns a new Tensor which indexes the original Tensor along dimension dim using the entries in torch.LongTensor index.
// The returned Tensor has the same number of dimensions as the original Tensor.
// The returned Tensor does not use the same storage as the original Tensor -- see below for storing the result in an existing Tensor.

// x = torch.rand(5,5)
// > x
// 0.8020  0.7246  0.1204  0.3419  0.4385
// 0.0369  0.4158  0.0985  0.3024  0.8186
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.7473  0.9028  0.1046  0.9085  0.6622
// 0.1412  0.6784  0.1624  0.8113  0.3949
// [torch.DoubleTensor of dimension 5x5]

// y = x:index(1,torch.LongTensor{3,1})
// > y
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.8020  0.7246  0.1204  0.3419  0.4385
// [torch.DoubleTensor of dimension 2x5]

// y:fill(1)
// > y
// 1  1  1  1  1
// 1  1  1  1  1
// [torch.DoubleTensor of dimension 2x5]

// > x
// 0.8020  0.7246  0.1204  0.3419  0.4385
// 0.0369  0.4158  0.0985  0.3024  0.8186
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.7473  0.9028  0.1046  0.9085  0.6622
// 0.1412  0.6784  0.1624  0.8113  0.3949
// [torch.DoubleTensor of dimension 5x5]
// Note the explicit index function is different than the indexing operator [].
// The indexing operator [] is a syntactic shortcut for a series of select and narrow operations,
// therefore it always returns a new view on the original tensor that shares the same storage.
// However, the explicit index function can not use the same storage.

// It is possible to store the result into an existing Tensor with result:index(source, ...):

// x = torch.rand(5,5)
// > x
// 0.8020  0.7246  0.1204  0.3419  0.4385
// 0.0369  0.4158  0.0985  0.3024  0.8186
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.7473  0.9028  0.1046  0.9085  0.6622
// 0.1412  0.6784  0.1624  0.8113  0.3949
// [torch.DoubleTensor of dimension 5x5]

// y = torch.Tensor()
// y:index(x,1,torch.LongTensor{3,1})
// > y
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.8020  0.7246  0.1204  0.3419  0.4385
// [torch.DoubleTensor of dimension 2x5]

// [Tensor] indexCopy(dim, index, tensor)

// Copies the elements of tensor into the original tensor by selecting the indices in the order given in index.
// The shape of tensor must exactly match the elements indexed or an error will be thrown.

// > x
// 0.8020  0.7246  0.1204  0.3419  0.4385
// 0.0369  0.4158  0.0985  0.3024  0.8186
// 0.2746  0.9362  0.2546  0.8586  0.6674
// 0.7473  0.9028  0.1046  0.9085  0.6622
// 0.1412  0.6784  0.1624  0.8113  0.3949
// [torch.DoubleTensor of dimension 5x5]

// z=torch.Tensor(5,2)
// z:select(2,1):fill(-1)
// z:select(2,2):fill(-2)
// > z
// -1 -2
// -1 -2
// -1 -2
// -1 -2
// -1 -2
// [torch.DoubleTensor of dimension 5x2]

// x:indexCopy(2,torch.LongTensor{5,1},z)
// > x
// -2.0000  0.7246  0.1204  0.3419 -1.0000
// -2.0000  0.4158  0.0985  0.3024 -1.0000
// -2.0000  0.9362  0.2546  0.8586 -1.0000
// -2.0000  0.9028  0.1046  0.9085 -1.0000
// -2.0000  0.6784  0.1624  0.8113 -1.0000
// [torch.DoubleTensor of dimension 5x5]

// [Tensor] indexAdd(dim, index, tensor)

// Accumulate the elements of tensor into the original tensor by adding to the indices in the order given in index.
// The shape of tensor must exactly match the elements indexed or an error will be thrown.

// Example 1

// > x
// -2.1742  0.5688 -1.0201  0.1383  1.0504
// 0.0970  0.2169  0.1324  0.9553 -1.9518
// -0.7607  0.8947  0.1658 -0.2181 -2.1237
// -1.4099  0.2342  0.4549  0.6316 -0.2608
// 0.0349  0.4713  0.0050  0.1677  0.2103
// [torch.DoubleTensor of size 5x5]

// z=torch.Tensor(5, 2)
// z:select(2,1):fill(-1)
// z:select(2,2):fill(-2)
// > z
// -1 -2
// -1 -2
// -1 -2
// -1 -2
// -1 -2
// [torch.DoubleTensor of dimension 5x2]

// > x:indexAdd(2,torch.LongTensor{5,1},z)
// > x
// -4.1742  0.5688 -1.0201  0.1383  0.0504
// -1.9030  0.2169  0.1324  0.9553 -2.9518
// -2.7607  0.8947  0.1658 -0.2181 -3.1237
// -3.4099  0.2342  0.4549  0.6316 -1.2608
// -1.9651  0.4713  0.0050  0.1677 -0.7897
// [torch.DoubleTensor of size 5x5]

// Example 2

// > a = torch.range(1, 5)
// > a
// 1
// 2
// 3
// 4
// 5
// [torch.DoubleTensor of size 5]

// > a:indexAdd(1, torch.LongTensor{1, 1, 3, 3}, torch.range(1, 4))
// > a
// 4
// 2
// 10
// 4
// 5
// [torch.DoubleTensor of size 5]

// [Tensor] indexFill(dim, index, val)

// Fills the elements of the original Tensor with value val by selecting the indices in the order given in index.

// x=torch.rand(5,5)
// > x
// 0.8414  0.4121  0.3934  0.5600  0.5403
// 0.3029  0.2040  0.7893  0.6079  0.6334
// 0.3743  0.1389  0.1573  0.1357  0.8460
// 0.2838  0.9925  0.0076  0.7220  0.5185
// 0.8739  0.6887  0.4271  0.0385  0.9116
// [torch.DoubleTensor of dimension 5x5]

// x:indexFill(2,torch.LongTensor{4,2},-10)
// > x
// 0.8414 -10.0000   0.3934 -10.0000   0.5403
// 0.3029 -10.0000   0.7893 -10.0000   0.6334
// 0.3743 -10.0000   0.1573 -10.0000   0.8460
// 0.2838 -10.0000   0.0076 -10.0000   0.5185
// 0.8739 -10.0000   0.4271 -10.0000   0.9116
// [torch.DoubleTensor of dimension 5x5]

// [Tensor] gather(dim, index)

// Creates a new Tensor from the original tensor by gathering a number of values from each "row",
// where the rows are along the dimension dim.
// The values in a LongTensor, passed as index, specify which values to take from each row.
// Specifically, the resulting Tensor, which will have the same size as the index tensor, is given by

// -- dim = 1
// result[i][j][k]... = src[index[i][j][k]...][j][k]...

// -- dim = 2
// result[i][j][k]... = src[i][index[i][j][k]...][k]...

// -- etc.
// where src is the original Tensor.

// The same number of values are selected from each row, and the same value cannot be selected from a row more than once.
// The values in the index tensor must not be larger than the length of the row, that is they must be between 1 and src:size(dim) inclusive.
// It can be somewhat confusing to ensure that the index tensor has the correct shape. Viewed pictorially:

// The gather operation

// Numerically, to give an example, if src has size n x m x p x q, we are gathering along dim = 3,
// and we wish to gather k elements from each row (where k <= p) then index must have size n x m x k x q.

// It is possible to store the result into an existing Tensor with result:gather(src, ...).

// x = torch.rand(5, 5)
// > x
// 0.7259  0.5291  0.4559  0.4367  0.4133
// 0.0513  0.4404  0.4741  0.0658  0.0653
// 0.3393  0.1735  0.6439  0.1011  0.7923
// 0.7606  0.5025  0.5706  0.7193  0.1572
// 0.1720  0.3546  0.8354  0.8339  0.3025
// [torch.DoubleTensor of size 5x5]

// y = x:gather(1, torch.LongTensor{{1, 2, 3, 4, 5}, {2, 3, 4, 5, 1}})
// > y
// 0.7259  0.4404  0.6439  0.7193  0.3025
// 0.0513  0.1735  0.5706  0.8339  0.4133
// [torch.DoubleTensor of size 2x5]

// z = x:gather(2, torch.LongTensor{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 1}})
// > z
// 0.7259  0.5291
// 0.4404  0.4741
// 0.6439  0.1011
// 0.7193  0.1572
// 0.3025  0.1720
// [torch.DoubleTensor of size 5x2]

// [Tensor] scatter(dim, index, src|val)

// Writes all values from tensor src or the scalar val into self at the specified indices.
// The indices are specified with respect to the given dimension, dim, in the manner described in gather. Note that, as for gather,
// the values of index must be between 1 and self:size(dim) inclusive and all values in a row along the specified dimension must be unique.

// x = torch.rand(2, 5)
// > x
// 0.3227  0.4294  0.8476  0.9414  0.1159
// 0.7338  0.5185  0.2947  0.0578  0.1273
// [torch.DoubleTensor of size 2x5]

// y = torch.zeros(3, 5):scatter(1, torch.LongTensor{{1, 2, 3, 1, 1}, {3, 1, 1, 2, 3}}, x)
// > y
// 0.3227  0.5185  0.2947  0.9414  0.1159
// 0.0000  0.4294  0.0000  0.0578  0.0000
// 0.7338  0.0000  0.8476  0.0000  0.1273
// [torch.DoubleTensor of size 3x5]

// z = torch.zeros(2, 4):scatter(2, torch.LongTensor{{3}, {4}}, 1.23)
// > z
// 0.0000  0.0000  1.2300  0.0000
// 0.0000  0.0000  0.0000  1.2300
// [torch.DoubleTensor of size 2x4]

// [Tensor] maskedSelect(mask)

// Returns a new Tensor which contains all elements aligned to a 1 in the corresponding mask.
// This mask is a torch.ByteTensor of zeros and ones. The mask and Tensor must have the same number of elements.
// The resulting Tensor will be a 1D tensor of the same type as Tensor having size mask:sum().

// x = torch.range(1,12):double():resize(3,4)
// > x
// 1   2   3   4
// 5   6   7   8
// 9  10  11  12
// [torch.DoubleTensor of dimension 3x4]

// mask = torch.ByteTensor(2,6):bernoulli()
// > mask
// 1  0  1  0  0  0
// 1  1  0  0  0  1
// [torch.ByteTensor of dimension 2x6]

// y = x:maskedSelect(mask)
// > y
// 1
// 3
// 7
// 8
// 12
// [torch.DoubleTensor of dimension 5]

// z = torch.DoubleTensor()
// z:maskedSelect(x, mask)
// > z
// 1
// 3
// 7
// 8
// 12
// Note how the dimensions of the above x, mask and y do not match. Also note how an existing tensor z can be used to store the results.


// [Tensor] maskedCopy(mask, tensor)

// Copies the masked elements of tensor into itself. The masked elements are those elements having a corresponding 1 in the mask Tensor.
// This mask is a torch.ByteTensor of zeros and ones. The destination Tensor and the mask Tensor should have the same number of elements.
// The source tensor should have at least as many elements as the number of 1s in the mask.

// x = torch.range(1,4):double():resize(2,2)
// > x
// 1  2
// 3  4
// [torch.DoubleTensor of dimension 2x4]

// mask = torch.ByteTensor(1,8):bernoulli()
// > mask
// 0  0  1  1  1  0  1  0
// [torch.ByteTensor of dimension 1x8]

// y = torch.DoubleTensor(2,4):fill(-1)
// > y
// -1 -1 -1 -1
// -1 -1 -1 -1
// [torch.DoubleTensor of dimension 2x4]

// y:maskedCopy(mask, x)
// > y
// -1 -1  1  2
// 3 -1  4 -1
// [torch.DoubleTensor of dimension 2x4]
// Note how the dimensions of the above x, mask and `y' do not match, but the number of elements do.


// [Tensor] maskedFill(mask, val)

// Fills the masked elements of itself with value val. The masked elements are those elements having a corresponding 1 in the mask Tensor.
// This mask is a torch.ByteTensor of zeros and ones. The mask and Tensor must have the same number of elements.

// x = torch.range(1,4):double():resize(1,4)
// > x
// 1  2  3  4
// [torch.DoubleTensor of dimension 1x4]

// mask = torch.ByteTensor(2,2):bernoulli()
// > mask
// 0  0
// 1  1
// [torch.ByteTensor of dimension 2x2]

// x:maskedFill(mask, -1)
// > x
// 1  2 -1 -1
// [torch.DoubleTensor of dimension 1x4]
// Note how the dimensions of the above x and mask do not match, but the number of elements do.

// Search

// Each of these methods returns a LongTensor corresponding to the indices of the given search operation.


// [LongTensor] nonzero(tensor)

// Finds and returns a LongTensor corresponding to the subscript indices of all non-zero elements in tensor.

// Note that torch uses the first argument on dispatch to determine the return type.
// Since the first argument is any torch.TensorType, but the return type is always torch.LongTensor, the function call torch.nonzero(torch.LongTensor(), tensor) does not work.
// However, tensor.nonzero(torch.LongTensor(), tensor) does work.

// > x = torch.rand(4, 4):mul(3):floor():int()
// > x
// 2  0  2  0
// 0  0  1  2
// 0  2  2  1
// 2  1  2  2
// [torch.IntTensor of dimension 4x4]

// > torch.nonzero(x)
// 1  1
// 1  3
// 2  3
// 2  4
// 3  2
// 3  3
// 3  4
// 4  1
// 4  2
// 4  3
// 4  4
// [torch.LongTensor of dimension 11x2]

// > x:nonzero()
// 1  1
// 1  3
// 2  3
// 2  4
// 3  2
// 3  3
// 3  4
// 4  1
// 4  2
// 4  3
// 4  4
// [torch.LongTensor of dimension 11x2]

// > indices = torch.LongTensor()
// > x.nonzero(indices, x)
// 1  1
// 1  3
// 2  3
// 2  4
// 3  2
// 3  3
// 3  4
// 4  1
// 4  2
// 4  3
// 4  4
// [torch.LongTensor of dimension 11x2]

// > x:eq(1):nonzero()
// 2  3
// 3  4
// 4  2
// [torch.LongTensor of dimension 3x2]

// Expanding/Replicating/Squeezing Tensors

// These methods returns a Tensor which is created by replications of the original tensor.


// [result] expand([result,] sizes)

// sizes can either be a torch.LongStorage or numbers. Expanding a tensor does not allocate new memory,
// but only creates a new view on the existing tensor where singleton dimensions can be expanded to multiple ones by setting the stride to 0.
// Any dimension that has size 1 can be expanded to arbitrary value without any new memory allocation.
// Attempting to expand along a dimension that does not have size 1 will result in an error.

// x = torch.rand(10,1)
// > x
// 0.3837
// 0.5966
// 0.0763
// 0.1896
// 0.4958
// 0.6841
// 0.4038
// 0.4068
// 0.1502
// 0.2239
// [torch.DoubleTensor of dimension 10x1]

// y = torch.expand(x,10,2)
// > y
// 0.3837  0.3837
// 0.5966  0.5966
// 0.0763  0.0763
// 0.1896  0.1896
// 0.4958  0.4958
// 0.6841  0.6841
// 0.4038  0.4038
// 0.4068  0.4068
// 0.1502  0.1502
// 0.2239  0.2239
// [torch.DoubleTensor of dimension 10x2]

// y:fill(1)
// > y
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// 1  1
// [torch.DoubleTensor of dimension 10x2]

// > x
// 1
// 1
// 1
// 1
// 1
// 1
// 1
// 1
// 1
// 1
// [torch.DoubleTensor of dimension 10x1]

    public Tensor expand(int... size) {
        Tensor t = new Tensor(this);
        if (t.size.length != size.length) throw new IllegalArgumentException("Wrong number of dimensions.");
        for (int i = 1; i <= size.length; ++i) {
            if (t.size[i - 1] == size[i - 1]) continue;
            if (t.size[i - 1] != 1)
                throw new IllegalArgumentException("Attempting to expand along a dimension that does not have size 1.");
            t.stride[i - 1] = 0;
            t.size[i - 1] = size[i - 1];
        }
        return t;
    }

// i=0; y:apply(function() i=i+1;return i end)
// > y
// 2   2
// 4   4
// 6   6
// 8   8
// 10  10
// 12  12
// 14  14
// 16  16
// 18  18
// 20  20
// [torch.DoubleTensor of dimension 10x2]

// > x
// 2
// 4
// 6
// 8
// 10
// 12
// 14
// 16
// 18
// 20
// [torch.DoubleTensor of dimension 10x1]

// [result] expandAs([result,] tensor)

// This is equivalent to self:expand(tensor:size())

    public Tensor expandAs(Tensor that) {
        return expand(that.size());
    }

// [Tensor] repeatTensor([result,] sizes)

// sizes can either be a torch.LongStorage or numbers.
// Repeating a tensor allocates new memory, unless result is provided, in which case its memory is resized.
// sizes specify the number of times the tensor is repeated in each dimension.

// x = torch.rand(5)
// > x
// 0.7160
// 0.6514
// 0.0704
// 0.7856
// 0.7452
// [torch.DoubleTensor of dimension 5]

// > torch.repeatTensor(x,3,2)
// 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
// 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
// 0.7160  0.6514  0.0704  0.7856  0.7452  0.7160  0.6514  0.0704  0.7856  0.7452
// [torch.DoubleTensor of dimension 3x10]

// > torch.repeatTensor(x,3,2,1)
// (1,.,.) =
// 0.7160  0.6514  0.0704  0.7856  0.7452
// 0.7160  0.6514  0.0704  0.7856  0.7452

// (2,.,.) =
// 0.7160  0.6514  0.0704  0.7856  0.7452
// 0.7160  0.6514  0.0704  0.7856  0.7452

// (3,.,.) =
// 0.7160  0.6514  0.0704  0.7856  0.7452
// 0.7160  0.6514  0.0704  0.7856  0.7452
// [torch.DoubleTensor of dimension 3x2x5]

// [Tensor] squeeze([dim])

// Removes all singleton dimensions of the tensor. If dim is given, squeezes only that particular dimension of the tensor.

// x=torch.rand(2,1,2,1,2)
// > x
// (1,1,1,.,.) =
// 0.6020  0.8897

// (2,1,1,.,.) =
// 0.4713  0.2645

// (1,1,2,.,.) =
// 0.4441  0.9792

// (2,1,2,.,.) =
// 0.5467  0.8648
// [torch.DoubleTensor of dimension 2x1x2x1x2]

// > torch.squeeze(x)
// (1,.,.) =
// 0.6020  0.8897
// 0.4441  0.9792

// (2,.,.) =
// 0.4713  0.2645
// 0.5467  0.8648
// [torch.DoubleTensor of dimension 2x2x2]

// > torch.squeeze(x,2)
// (1,1,.,.) =
// 0.6020  0.8897

// (2,1,.,.) =
// 0.4713  0.2645

// (1,2,.,.) =
// 0.4441  0.9792

// (2,2,.,.) =
// 0.5467  0.8648
// [torch.DoubleTensor of dimension 2x2x1x2]
// Manipulating the tensor view

// Each of these methods returns a Tensor which is another way of viewing the Storage of the given tensor.
// Hence, any modification in the memory of the sub-tensor will have an impact on the primary tensor, and vice-versa.

// These methods are very fast, because they do not involve any memory copy.

    public Tensor squeeze(int... dim) {
        if (dim.length == 0) {
            dim = new int[size.length];
            for (int i = 0; i < size.length; ++i) dim[i] = i + 1;
        } else {
            java.util.Arrays.sort(dim);
        }
        Tensor t = new Tensor(this);
        for (int i = dim.length - 1; i >= 0; --i) {
            if (t.size(dim[i]) == 1) {
                t = t.select(dim[i], 1);
            }
        }
        return t;
    }


// [result] view([result,] tensor, sizes)

// Creates a view with different dimensions of the storage associated with tensor. 
// If result is not passed, then a new tensor is returned, 
// otherwise its storage is made to point to storage of tensor.

// sizes can either be a torch.LongStorage or numbers. 
// If one of the dimensions is -1, the size of that dimension is inferred from the rest of the elements.

// x = torch.zeros(4)
// > x:view(2,2)
// 0 0
// 0 0
// [torch.DoubleTensor of dimension 2x2]

// > x:view(2,-1)
// 0 0
// 0 0
// [torch.DoubleTensor of dimension 2x2]

// > x:view(torch.LongStorage{2,2})
// 0 0
// 0 0
// [torch.DoubleTensor of dimension 2x2]

// > x
// 0
// 0
// 0
// 0
// [torch.DoubleTensor of dimension 4]

    public Tensor view(int... size) {
        if (!isContiguous()) throw new IllegalArgumentException("Expecting a contiguous tensor.");
        _fixDefaultSize(size, _totalSize(this.size));
        Tensor t = new Tensor(this);
        t.size = _copyIntArray(size);
        t.stride = _defaultStride(size);
        t.storageOffset = new int[size.length + 1];
        t.storageOffset[0] = this.storageOffset[0]; // For contiguous tensor, all other storageOffsets are 0's
        t.dimPermutation = null;
        return t;
    }

    static void _fixDefaultSize(int[] size, int totalSize) {
        int unknownIndex = -1;
        for (int i = 0; i < size.length; ++i) {
            if (size[i] != -1) continue;
            unknownIndex = i;
            size[i] = 1;
            break;
        }
        int totalSize2 = _totalSize(size);
        if (unknownIndex != -1) {
            size[unknownIndex] = totalSize / totalSize2;
            totalSize2 *= size[unknownIndex];
        }
        if (totalSize2 != totalSize) throw new IllegalArgumentException("Wrong size for view.");
    }


// [result] viewAs([result,] tensor, template)

// Creates a view with the same dimensions as template of the storage associated with tensor.
// If result is not passed, then a new tensor is returned, otherwise its storage is made to point to storage of tensor.

// x = torch.zeros(4)
// y = torch.Tensor(2,2)
// > x:viewAs(y)
// 0 0
// 0 0
// [torch.DoubleTensor of dimension 2x2]

    public Tensor viewAs(Tensor that) {
        return view(that.size());
    }

// [Tensor] transpose(dim1, dim2)

// Returns a tensor where dimensions dim1 and dim2 have been swapped. For 2D tensors, the convenience method of t() is available.

// x = torch.Tensor(3,4):zero()
// x:select(2,3):fill(7) -- fill column 3 with 7
// > x
// 0  0  7  0
// 0  0  7  0
// 0  0  7  0
// [torch.DoubleTensor of dimension 3x4]

// y = x:transpose(1,2) -- swap dimension 1 and 2
// > y
// 0  0  0
// 0  0  0
// 7  7  7
// 0  0  0
// [torch.DoubleTensor of dimension 4x3]

// y:select(2, 3):fill(8) -- fill column 3 with 8
// > y
// 0  0  8
// 0  0  8
// 7  7  8
// 0  0  8
// [torch.DoubleTensor of dimension 4x3]

// > x -- contents of x have changed as well
// 0  0  7  0
// 0  0  7  0
// 8  8  8  8
// [torch.DoubleTensor of dimension 3x4]

    public Tensor transpose(int dim1, int dim2) {
        Tensor t = new Tensor(this);
        if (t.dimPermutation == null) t.dimPermutation = _intRange(1, t.size.length, 1);
        _swapIntArray(t.dimPermutation, dim1 - 1, dim2 - 1);
        t.dimPermutation = _validateDimPermutation(t.dimPermutation);
        return t;
    }

    static int[] _intRange(int start, int end, int interval) {
        int count = (end - start) / interval + 1;
        int[] range = new int[count];
        for (int i = 0; i < count; ++i) {
            range[i] = start + i * interval;
        }
        return range;
    }

    static void _swapIntArray(int[] array, int index1, int index2) {
        int temp = array[index1];
        array[index1] = array[index2];
        array[index2] = temp;
    }

    static int[] _validateDimPermutation(int[] dimPermutation) {
        if (dimPermutation == null) return null;
        int[] range = _intRange(1, dimPermutation.length, 1);
        if (_compareIntArrays(range, dimPermutation)) return null;
        int[] dimPermutation2 = _copyIntArray(dimPermutation);
        java.util.Arrays.sort(dimPermutation2);
        if (!_compareIntArrays(dimPermutation2, range)) throw new IllegalArgumentException("Invalid permutation.");
        return dimPermutation;
    }

// [Tensor] t()

// Convenience method of transpose() for 2D tensors. The given tensor must be 2 dimensional. Swap dimensions 1 and 2.

// x = torch.Tensor(3,4):zero()
// x:select(2,3):fill(7)
// y = x:t()
// > y
// 0  0  0
// 0  0  0
// 7  7  7
// 0  0  0
// [torch.DoubleTensor of dimension 4x3]

// > x
// 0  0  7  0
// 0  0  7  0
// 0  0  7  0
// [torch.DoubleTensor of dimension 3x4]

    public Tensor t() {
        if (size.length != 2) throw new IllegalArgumentException("Tensor must have 2 dimensions.");
        return transpose(1, 2);
    }

// [Tensor] permute(dim1, dim2, ..., dimn)

// Generalizes the function transpose() and can be used as a convenience method replacing a sequence of transpose() calls.
// Returns a tensor where the dimensions were permuted according to the permutation given by (dim1, dim2, ... , dimn).
// The permutation must be specified fully, i.e. there must be as many parameters as the tensor has dimensions.

// x = torch.Tensor(3,4,2,5)
// > x:size()
// 3
// 4
// 2
// 5
// [torch.LongStorage of size 4]

// y = x:permute(2,3,1,4) -- equivalent to y = x:transpose(1,3):transpose(1,2)
// > y:size()
// 4
// 2
// 3
// 5
// [torch.LongStorage of size 4]

    public Tensor permute(int... permutation) {
        Tensor t = new Tensor(this);
        if (t.dimPermutation == null) t.dimPermutation = _intRange(1, t.size.length, 1);
        t.dimPermutation = _physicalDimension(permutation, t.dimPermutation);
        t.dimPermutation = _validateDimPermutation(t.dimPermutation);
        return t;
    }

// [Tensor] unfold(dim, size, step)

// Returns a tensor which contains all slices of size size in the dimension dim. 
// Step between two slices is given by step.

// If sizedim is the original size of dimension dim, 
// the size of dimension dim in the returned tensor will be (sizedim - size) / step + 1

// An additional dimension of size size is appended in the returned tensor.

// x = torch.Tensor(7)
// for i=1,7 do x[i] = i end
// > x
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// [torch.DoubleTensor of dimension 7]

// > x:unfold(1, 2, 1)
// 1  2
// 2  3
// 3  4
// 4  5
// 5  6
// 6  7
// [torch.DoubleTensor of dimension 6x2]

// > x:unfold(1, 2, 2)
// 1  2
// 3  4
// 5  6
// [torch.DoubleTensor of dimension 3x2]
// Applying a function to a tensor

    public Tensor unfold(int dim, int size, int step) {
        Tensor t = narrow(dim, 1, size);

        int dim0 = dim;
        dim = _physicalDimension(dimPermutation, dim);

        t.storageOffset = _insertIntArray(t.storageOffset, dim + 1, 0);
        int size1 = (this.size[dim - 1] - size) / step + 1;
        t.size = _insertIntArray(t.size, dim, size1);
        t.size[dim] = size;
        t.stride = _insertIntArray(t.stride, dim, t.stride[dim - 1] * step);
        t.dimPermutation = _appendPermutation(t.dimPermutation, dim0, t.size.length - 1);
        return t;
    }

    static int[] _insertIntArray(int[] array, int dim, int value) {
        int[] array2 = new int[array.length + 1];
        _copyIntArray(array2, array, 0, dim - 1, 0);
        _copyIntArray(array2, array, dim - 1, array.length, 1);
        array2[dim - 1] = value;
        return array2;
    }

    static int[] _appendPermutation(int[] dimPermutation, int dim0, int size) {
        if (dimPermutation == null) dimPermutation = _intRange(1, size, 1);
        dimPermutation = _insertIntArray(dimPermutation, size + 1, 0);
        int dim = dimPermutation[dim0 - 1];
        for (int i = 0; i < dimPermutation.length; ++i) {
            if (dimPermutation[i] > dim) ++dimPermutation[i];
        }
        dimPermutation[size] = dim + 1;
        return _validateDimPermutation(dimPermutation);
    }

// These functions apply a function to each element of the tensor on which called the method (self). 
// These methods are much faster than using a for loop in Lua. 
// The results is stored in self (if the function returns something).

// [self] apply(function)

// Apply the given function to all elements of self.

// The function takes a number (the current element of the tensor) and might return a number, in which case it will be stored in self.

// Examples:

// i = 0
// z = torch.Tensor(3,3)
// z:apply(function(x)
// i = i + 1
// return i
// end) -- fill up the tensor
// > z
// 1  2  3
// 4  5  6
// 7  8  9
// [torch.DoubleTensor of dimension 3x3]

// z:apply(math.sin) -- apply the sin function
// > z
// 0.8415  0.9093  0.1411
// -0.7568 -0.9589 -0.2794
// 0.6570  0.9894  0.4121
// [torch.DoubleTensor of dimension 3x3]

// sum = 0
// z:apply(function(x)
// sum = sum + x
// end) -- compute the sum of the elements
// > sum
// 1.9552094821074

// > z:sum() -- it is indeed correct!
// 1.9552094821074

// [self] map(tensor, function(xs, xt))

// Apply the given function to all elements of self and tensor. The number of elements of both tensors must match, but sizes do not matter.

// The function takes two numbers (the current element of self and tensor) and might return a number, in which case it will be stored in self.

// Example:

// x = torch.Tensor(3,3)
// y = torch.Tensor(9)
// i = 0
// x:apply(function() i = i + 1; return i end) -- fill-up x
// i = 0
// y:apply(function() i = i + 1; return i end) -- fill-up y
// > x
// 1  2  3
// 4  5  6
// 7  8  9
// [torch.DoubleTensor of dimension 3x3]

// > y
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// 8
// 9
// [torch.DoubleTensor of dimension 9]

// x:map(y, function(xx, yy) return xx*yy end) -- element-wise multiplication
// > x
// 1   4   9
// 16  25  36
// 49  64  81
// [torch.DoubleTensor of dimension 3x3]

    @FunctionalInterface
    interface Function2<T1, T2, R> {
        public R apply(T1 t1, T2 t2);
    }

    public Tensor map(Tensor y, Function2<Double, Double, Double> function2) {
        int nel = nElement();
        if (nel != y.nElement()) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        TensorIndexesIterator iity = new TensorIndexesIterator(y);
        for (int i = 0; i < nel; ++i) {
            int[] ix = iitx.next();
            double x1 = get(ix);
            double y1 = y.get(iity.next());
            set(function2.apply(x1, y1), ix);
        }
        return this;
    }

    public Tensor map(Tensor y, java.util.function.Function<Double, Double> function) {
        int nel = nElement();
        if (nel != y.nElement()) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        TensorIndexesIterator iity = new TensorIndexesIterator(y);
        for (int i = 0; i < nel; ++i) {
            double y1 = y.get(iity.next());
            set(function.apply(y1), iitx.next());
        }
        return this;
    }

// [self] map2(tensor1, tensor2, function(x, xt1, xt2))

// Apply the given function to all elements of self, tensor1 and tensor2. The number of elements of all tensors must match, but sizes do not matter.

// The function takes three numbers (the current element of self, tensor1 and tensor2) and might return a number, in which case it will be stored in self.

// Example:

// x = torch.Tensor(3,3)
// y = torch.Tensor(9)
// z = torch.Tensor(3,3)

// i = 0; x:apply(function() i = i + 1; return math.cos(i)*math.cos(i) end)
// i = 0; y:apply(function() i = i + 1; return i end)
// i = 0; z:apply(function() i = i + 1; return i end)

// > x
// 0.2919  0.1732  0.9801
// 0.4272  0.0805  0.9219
// 0.5684  0.0212  0.8302
// [torch.DoubleTensor of dimension 3x3]

// > y
// 1
// 2
// 3
// 4
// 5
// 6
// 7
// 8
// 9
// [torch.DoubleTensor of dimension 9]

// > z
// 1  2  3
// 4  5  6
// 7  8  9
// [torch.DoubleTensor of dimension 3x3]

// x:map2(y, z, function(xx, yy, zz) return xx+yy*zz end)
// > x
// 1.2919   4.1732   9.9801
// 16.4272  25.0805  36.9219
// 49.5684  64.0212  81.8302
// [torch.DoubleTensor of dimension 3x3]

    @FunctionalInterface
    interface Function3<T1, T2, T3, R> {
        public R apply(T1 t1, T2 t2, T3 t3);
    }

    public Tensor map2(Tensor y, Tensor z, Function3<Double, Double, Double, Double> function3) {
        int nel = nElement();
        if (nel != y.nElement() || nel != z.nElement()) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        TensorIndexesIterator iity = new TensorIndexesIterator(y);
        TensorIndexesIterator iitz = new TensorIndexesIterator(z);
        for (int i = 0; i < nel; ++i) {
            int[] ix = iitx.next();
            double x1 = get(ix);
            double y1 = y.get(iity.next());
            double z1 = z.get(iitz.next());
            set(function3.apply(x1, y1, z1), ix);
        }
        return this;
    }

    public Tensor map2(Tensor y, Tensor z, Function2<Double, Double, Double> function2) {
        int nel = nElement();
        if (nel != y.nElement() || nel != z.nElement()) {
            throw new IllegalArgumentException("Number of elements do not agree.");
        }
        TensorIndexesIterator iitx = new TensorIndexesIterator(this);
        TensorIndexesIterator iity = new TensorIndexesIterator(y);
        TensorIndexesIterator iitz = new TensorIndexesIterator(z);
        for (int i = 0; i < nel; ++i) {
            double y1 = y.get(iity.next());
            double z1 = z.get(iitz.next());
            set(function2.apply(y1, z1), iitx.next());
        }
        return this;
    }


    static class TensorIndexesIterator {
        private int[] size;
        private int[] indexes;

        public TensorIndexesIterator(Tensor t) {
            size = t.size();
            indexes = new int[size.length];
            for (int i = 0; i < size.length - 1; ++i) indexes[i] = 1;
        }

        public int[] next() {
            ++indexes[size.length - 1];
            for (int i = size.length - 1; i >= 0; --i) {
                if (indexes[i] > size[i]) {
                    indexes[i] = 1;
                    if (i == 0) throw new java.util.NoSuchElementException();
                    ++indexes[i - 1];
                }
            }
            return indexes;
        }
    }


// Dividing a tensor into a table of tensors

// These functions divide a Tensor into a table of Tensors.


// [result] split([result,] tensor, size, [dim])

// Splits Tensor tensor along dimension dim into a result table of Tensors of size size (a number) or less (in the case of the last Tensor).
// The sizes of the non-dim dimensions remain unchanged. Internally, a series of narrows are performed along dimensions dim. Argument dim defaults to 1.

// If result is not passed, then a new table is returned, otherwise it is emptied and reused.

// Example:

// x = torch.randn(3,4,5)

// > x:split(2,1)
// {
// 1 : DoubleTensor - size: 2x4x5
// 2 : DoubleTensor - size: 1x4x5
// }

// > x:split(3,2)
// {
// 1 : DoubleTensor - size: 3x3x5
// 2 : DoubleTensor - size: 3x1x5
// }

// > x:split(2,3)
// {
// 1 : DoubleTensor - size: 3x4x2
// 2 : DoubleTensor - size: 3x4x2
// 3 : DoubleTensor - size: 3x4x1
// }

    public Tensor[] split(int size, int dim) {
        int totalSize = size(dim);
        int count = (int) Math.ceil(totalSize / (double) size);
        Tensor[] tensors = new Tensor[count];
        for (int i = 0; i < tensors.length; ++i) {
            tensors[i] = narrow(dim, i * size + 1, (totalSize > size ? size : totalSize));
            totalSize -= size;
        }
        return tensors;
    }


// [result] chunk([result,] tensor, n, [dim])

// Splits Tensor tensor into n chunks of approximately equal size along dimensions dim and returns these as a result table of Tensors. Argument dim defaults to 1.

// This function uses split internally:

// torch.split(result, tensor, math.ceil(tensor:size(dim)/n), dim)
// Example:

// x = torch.randn(3,4,5)

// > x:chunk(2,1)
// {
// 1 : DoubleTensor - size: 2x4x5
// 2 : DoubleTensor - size: 1x4x5
// }

// > x:chunk(2,2)
// {
// 1 : DoubleTensor - size: 3x2x5
// 2 : DoubleTensor - size: 3x2x5
// }

// > x:chunk(2,3)
// {
// 1 : DoubleTensor - size: 3x4x3
// 2 : DoubleTensor - size: 3x4x2
// }

    public Tensor[] chunk(int n, int dim) {
        int size = (int) Math.ceil(size(dim) / (double) n);
        return split(size, dim);
    }

// LuaJIT FFI access

// These functions expose Torch's Tensor and Storage data structures, through LuaJIT FFI. This allows extremely fast access to Tensors and Storages, all from Lua.


// [result] data(tensor, [asnumber])

// Returns a LuaJIT FFI pointer to the raw data of the tensor. If asnumber is true, then returns the pointer as a intptr_t cdata that you can transform to a plain lua number with tonumber().

// Accessing the raw data of a Tensor like this is extremely efficient, in fact, it's almost as fast as C in lots of cases.

// Example:

// t = torch.randn(3,2)
// > t
// 0.8008 -0.6103
// 0.6473 -0.1870
// -0.0023 -0.4902
// [torch.DoubleTensor of dimension 3x2]

// t_data = torch.data(t)
// for i = 0,t:nElement()-1 do t_data[i] = 0 end
// > t
// 0 0
// 0 0
// 0 0
// [torch.DoubleTensor of dimension 3x2]
// WARNING: bear in mind that accessing the raw data like this is dangerous, and should only be done on contiguous tensors (if a tensor is not contiguous,
// then you have to use its size and stride information). Making sure a tensor is contiguous is easy:

// t = torch.randn(3,2)
// t_noncontiguous = t:transpose(1,2)

// -- it would be unsafe to work with torch.data(t_noncontiguous)
// t_transposed_and_contiguous = t_noncontiguous:contiguous()

// -- it is now safe to work with the raw pointer
// data = torch.data(t_transposed_and_contiguous)
// Last, the pointer can be returned as a plain intptr_t cdata.
// This can be useful to share pointers between threads (warning: this is dangerous, as the second tensor doesn't increment the reference counter on the storage.
// If the first tensor gets freed, then the data of the second tensor becomes a dangling pointer):

// t = torch.randn(10)
// p = tonumber(torch.data(t,true))
// s = torch.Storage(10, p)
// tt = torch.Tensor(s)
// -- tt and t are a view on the same data.

// [result] cdata(tensor, [asnumber])

// Returns a LuaJIT FFI pointer to the C structure of the tensor. Use this with caution, and look at FFI.lua for the members of the tensor

// Reference counting

// Tensors are reference-counted. It means that each time an object (C or the Lua state) need to keep a reference over a tensor,
// the corresponding tensor reference counter will be increased. The reference counter is decreased when the object does not need the tensor anymore.

// These methods should be used with extreme care. In general, they should never be called, except if you know what you are doing,
// as the handling of references is done automatically. They can be useful in threaded environments. Note that these methods are atomic operations.


// retain()

// Increment the reference counter of the tensor.


// free()

// Decrement the reference counter of the tensor. Free the tensor if the counter is at 0.
// Status API Training Shop Blog About Pricing
// © 2015 GitHub, Inc. Terms Privacy Security Contact Help

}