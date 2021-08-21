<TeXmacs|1.99.18>

<style|generic>

<\body>
  <doc-data|<doc-title|Project Report>|<doc-subtitle|Architecture and
  Platforms for AI (M2)>|<doc-author|<author-data|<author-name|Hanying
  Zhang>|<\author-affiliation>
    July, 2021
  </author-affiliation>>>>

  <section|Brief Problem Description>

  In this project we implement a simplified Neural Network. For each layer,
  the difference with a standard fully connected layer is that, when
  computing the value of each neuron node in the output layer, instead of
  using all neurons in the input layer, we use only R neurons, as shown in
  the left figure below. The activation function we use here is Sigmoid.

  We then combine K such layers to form a Neural Network. The W and b
  parameters of each layer may be different while R is the same for all
  layers.\ 

  <image|problem-layer.png|97pt|115pt|500|><image|problem-all.png|179pt|114pt|500|0>

  <section|Machine Specific>

  The whole project was written and tested on my local machine. The CPU is a
  i7-6700HQ with 4 physical cores. It also uses <em|Intel>'s
  <em|Hyper-threading> technology which need to be turned off because the
  performance of the virtual cores is only 20% of the phisycal cores on
  average.

  The GPU is a Nvidia GTX 1060 with 6 GB global memory. The specification is
  shown in the image below:

  <\big-figure|>
    GTX 1060 specification
  </big-figure>

  The memory of this machine is 16 GB xxHz dual channels.

  The operating system is Ubuntu 20.04.1 LTS. The GCC version is . The CUDA
  version, NVCC version .

  <section|Serial Implementation>

  The maim purpose of the serial implemention is to check the correctness of
  the OpenMP and CUDA implementations.

  The serial implementation is very straitforward. In the main function, I
  loop over all K layers. For each layer, I wrote a function to perform the
  culculation in which I loop over all the neurons in the output layer. For
  each neuron, I loop over the R MACs.

  <section|The Memory Management Evolution>

  In the first versions of both the OpenMP and CUDA implementations, I used
  one array in the stack to store the latest layer result got calculated. I
  re-used the <em|N-t(R-1)> elements from the beginning of the array when the
  <em|t-th> layer got the result. The benefit of this method is the small
  memory costage. But it is very different from the real life implementation
  of Neural Networks. For instance, if, in future, we want to extend our
  project to be able to do the back propagation, then we will be lack of
  information.

  So I changed the memory management method by storing all the results of all
  layers in the heap. In this method we consumed more memory but we also kept
  the necessary information.

  The second benefit of this method is that we reduces the number of times
  calling CUDAMemcpy. In the first version, we need to call CudaMemcpy
  <em|K-1> times because for each layer we need to do that. In the second
  version, we can do that once for all. As the CudaMemcpy costs lots of time,
  the total time costage using the second version could be much smaller.

  The performance of this method is not good, cudaMemcpy cost lots of time.

  So the idea is to reduce the times of the mem cpy between host and device.

  <section|OpenMP Implementation>

  The OpenMP implementation is very similar to the serial one.

  TODO: only use FOR seems very little knowledge about OpenMP. Schedule?
  Reduction?

  Introduce how to get the parameters of N and K from terminal

  <section|CUDA Implementation>

  When doing the CUDA implementation, the first idea came into mind is using
  one block to calculate one neuron in the output layer. The problem with
  this method is that most threads in each block are wasted if R is a small
  value. TODO: smaller than 32?\ 

  So using one block to calculate more than one neuron is a better idea. In
  this way we can also use the shared memory in each block.

  <section|Correctness Checking>

  The method we use is to make sure that the OpenMP version and CUDA version
  could get the same result as the serial version.

  In order to do this, we must make sure that the initial values of the
  parameters are all the same in these 3 versions. I use a random seed to
  make sure the random values got are the same.\ 

  I also manully calculated the result of N=7 and K=3 to make sure that the
  serial implementation is also correctly calculated.

  <section|Performance Analysis>

  In order to get the accurate result as much as possible, I switched off all
  other applications not necessary. For each time costage, I run the code for
  5 times and used their mean result.

  <subsection|OpenMP>

  <subsubsection|Strong Scaling>

  <subsubsection|Weak Scaling>

  <subsection|CUDA>

  <subsubsection|Strong Scaling>

  <subsubsection|Weak Scaling>

  \;

  \;
</body>

<\initial>
  <\collection>
    <associate|page-medium|paper>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-10|<tuple|8.1|?>>
    <associate|auto-11|<tuple|8.1.1|?>>
    <associate|auto-12|<tuple|8.1.2|?>>
    <associate|auto-13|<tuple|8.2|?>>
    <associate|auto-14|<tuple|8.2.1|?>>
    <associate|auto-15|<tuple|8.2.2|?>>
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|1|1>>
    <associate|auto-4|<tuple|3|1>>
    <associate|auto-5|<tuple|4|1>>
    <associate|auto-6|<tuple|5|2>>
    <associate|auto-7|<tuple|6|2>>
    <associate|auto-8|<tuple|7|?>>
    <associate|auto-9|<tuple|8|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Problem
      Brief Description> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Serial
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>OpenMP
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>CUDA
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Correctness
      Checking> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Performance
      Analysis> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|7<space|2spc>Machine
      Specific> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>