<TeXmacs|1.99.19>

<style|<tuple|generic|chinese>>

<\body>
  <doc-data|<doc-title|Project Report>|<doc-subtitle|Architecture and
  Platforms for AI (M2)>|<doc-author|<author-data|<author-name|Hanying
  Zhang>|<\author-affiliation>
    July, 2021
  </author-affiliation>>>>

  <section|Problem Description>

  <subsection|The variables and the main problem constraints>

  We wrote a python script file named <em|txt2dzn.py> whose function is
  converting the instance files from <em|text> file format to <em|dzn> file
  format. We then read in <em|W, H, N, Ws> and <em|Hs> respectivaly as the
  width of the whole paper, the height of the whole paper, the number of the
  small pieces, the widths of the small pieces and the heights of the small
  pieces. We created two decision variables <em|Xs> and <em|Ys>, which
  represent the coordinates of the left-bottom corner of all the small
  pieces.\ 

  <subsection|Implied constraints>

  Take a vertical line for instance, the total heights of all the traversed
  pieces should not be bigger than <math|H>, and we should check all the
  vertical lines. For the CP problem we use <em|sum> and <em|forall> for
  these two requirements respectively.\ 

  <section|Openmp Implementation>

  <subsection|Auxilliary constraints>

  <subsubsection|The small pieces whose heights <em|Hs[i] \<gtr\> 0.5 * H>
  could only be placed horizontally>

  If the total heights of 2 rectangles are both bigger than half the height
  of the whole sheet, then they could not be placed vertically, which means
  that the <math|x> coordinates are different. Thus we could use global
  constrint <em|alldifferent> upon them. More strictly, suppose their width
  are <math|w<rsub|i> and w<rsub|j> respectively,>and\ 

  <section|Cuda Implementation>

  <section|Correctness Checking>

  \;

  <section|Performance Analysis>

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
    <associate|auto-2|<tuple|1.1|1>>
    <associate|auto-3|<tuple|1.2|1>>
    <associate|auto-4|<tuple|2|1>>
    <associate|auto-5|<tuple|2.1|1>>
    <associate|auto-6|<tuple|2.1.1|1>>
    <associate|auto-7|<tuple|3|1>>
    <associate|auto-8|<tuple|4|1>>
    <associate|auto-9|<tuple|5|1>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Problem
      Description> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1tab>|1.1<space|2spc>The variables and the main
      problem constraints <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1tab>|1.2<space|2spc>Implied constraints
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Openmp
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4><vspace|0.5fn>

      <with|par-left|<quote|1tab>|2.1<space|2spc>Auxilliary constraints
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|2tab>|2.1.1<space|2spc>The small pieces whose
      heights <with|font-shape|<quote|italic>|Hs[i] \<gtr\> 0.5 * H> could
      only be placed horizontally <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|3<space|2spc>Cuda
      Implementation> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|4<space|2spc>Correctness
      Checking> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|5<space|2spc>Performance
      Analysis> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>