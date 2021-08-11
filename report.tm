<TeXmacs|1.99.19>

<style|generic>

<\body>
  <doc-data|<doc-title|Project Report>|<doc-subtitle|Architecture and
  Platforms for AI (M2)>|<doc-author|<author-data|<author-name|Hanying
  Zhang>|<\author-affiliation>
    July, 2021
  </author-affiliation>>>>

  <section|Problem Brief Description>

  In this project we implement a simplified Neural Network. For each layer,
  the difference with a standard fully connected layer is that, when
  computing the value of each neuron node in the output layer, instead of
  using all neurons in the input layer, we use only R neurons, as shown in
  the left figure below. The activation function we use here is Sigmoid.

  We then combine K such layers to form a Neural Network. The W and b
  parameters of each layer may be different while R is the same for all
  layers.\ 

  <image|problem-layer.png|97pt|115pt|50|><image|problem-all.png|179pt|114pt|100|>

  <section|Serial Implementation>

  The serial implementation of this NN is very straitforward. In the main
  function, we loop over all K layers. For each layer, I wrote a function to
  perform the culculation. In this function, we loop over all the neurons in
  the output layer.

  <section|OpenMP Implementation>

  The OpenMP implementation is very similar with the serial one.

  <section|CUDA Implementation>

  When doing CUDA implementation, the first idea came into mind is using one
  block to calculate one neuron in the output layer. The problem with this
  method is that<fill-dots>\ 

  So using one block to calculate more than one neuron is a better idea. and
  we can use the shared memory in each block.

  The performance of this method is not good, cudaMemcpy cost lots of time.

  So the idea is to reduce the times of the mem cpy between host and device.

  <section|Correctness Checking>

  the method we use is to make sure the OpenMP version and CUDA version could
  get the same result as the serial version.

  In order to do this, we must make sure that the initial values of the
  parameters are all the same in these 3 versions.

  I use a random seed to make sure the random values got are the same.\ 

  <section|Performance Analysis>

  <section|Machine Specific>

  I used\ 

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
    <associate|auto-2|<tuple|2|1>>
    <associate|auto-3|<tuple|3|1>>
    <associate|auto-4|<tuple|4|1>>
    <associate|auto-5|<tuple|5|1>>
    <associate|auto-6|<tuple|6|1>>
    <associate|auto-7|<tuple|7|1>>
    <associate|auto-8|<tuple|6|1>>
    <associate|auto-9|<tuple|7|?>>
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

      <with|par-left|<quote|1tab>|3.1<space|2spc>Auxilliary constraints
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|2tab>|3.1.1<space|2spc>The small pieces whose
      heights <with|font-shape|<quote|italic>|Hs[i] \<gtr\> 0.5 * H> could
      only be placed horizontally <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>CUDA
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Correctness
      Checking> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|6<space|2spc>Performance
      Analysis> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>