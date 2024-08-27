<h1>TADMA neural network inference library</h1>
<h2> *** Work in progress *** </h2>



<h2> Requirements </h2>
<ul>
<li> Clang >= 18 and C++>=23 </li>
<li> CUDA </li>
<li> cuBLASLt </li>
</ul>


Desired final structure

1. Top level python driver library, accepts .onnx file and provides an interface to do inference
2. Generate a C++ library which loads, stores and provides access to the weights based on the onnx file
3. Generate a .cu file which uses the utilities in include/tadma to implement an inference engine
4. Every time the python library is called with a unique input shape (ex. batch size, seq len), generate a new shared library <br> from the .cu file using clang++ and call it with the weights from (2.) to do inference
5. Return results to the user